import modal
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLControlNetPipeline,
    DPMSolverMultistepScheduler,
    AutoencoderKL,
    ControlNetModel,
)
from PIL import Image
import numpy as np
import base64
import io
import torch
from transformers import DPTFeatureExtractor, DPTForDepthEstimation
from controlnet_aux.open_pose import OpenposeDetector
import cv2

app = modal.App("multi-model-image-api")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgl1", "libglib2.0-0", "libsm6", "libxext6")
    .pip_install(
        "numpy<2.0",
        "torch==2.1.2",
        "diffusers==0.26.3",
        "transformers==4.38.2",
        "accelerate==0.27.2",
        "opencv-python==4.9.0.80",
        "safetensors==0.4.2",
        "Pillow==10.2.0",
        "fastapi==0.109.1",
        "pydantic==2.6.4",
        "mediapipe==0.10.9",
        "controlnet-aux==0.0.7",
        "scikit-image==0.22.0",
        "huggingface-hub==0.20.3"
    )
)

MODELS = {
    "base": "stabilityai/stable-diffusion-xl-base-1.0",
    "refiner": "stabilityai/stable-diffusion-xl-refiner-1.0",
    "depth": "diffusers/controlnet-depth-sdxl-1.0",
    "pose": "thibaud/controlnet-openpose-sdxl-1.0",  # Updated to working SDXL model
    "canny": "diffusers/controlnet-canny-sdxl-1.0",
    "vae": "madebyollin/sdxl-vae-fp16-fix"
}

def decode_img(b64: str) -> Image.Image:
    try:
        b64 = b64.strip()
        if len(b64) % 4 != 0:
            b64 += '=' * (4 - len(b64) % 4)
        img_data = base64.b64decode(b64)
        img = Image.open(io.BytesIO(img_data))
        return img.convert("RGB").resize((1024, 1024))
    except Exception as e:
        print(f"Image decoding failed: {str(e)}")
        return Image.new("RGB", (1024, 1024), "white")

def encode_img(img: Image.Image) -> str:
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def get_controlnet_output(prompt, guide_img, model_id, vae, seed):
    try:
        # Verify model is SDXL compatible
        if "sdxl" not in model_id.lower() and "xl" not in model_id.lower():
            raise ValueError(f"Model {model_id} is not SDXL compatible")
            
        controlnet = ControlNetModel.from_pretrained(model_id, torch_dtype=torch.float16)
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            MODELS["base"],
            controlnet=controlnet,
            torch_dtype=torch.float16,
            vae=vae
        )
        pipe.enable_model_cpu_offload()
        result = pipe(
            prompt=prompt,
            image=guide_img,
            num_inference_steps=30,
            guidance_scale=8.0,
            generator=torch.manual_seed(seed)
        ).images[0]
        return encode_img(result)
    except Exception as e:
        print(f"ControlNet failed for {model_id}: {str(e)}")
        return "generation_failed"

@app.function(image=image, gpu="A100-40GB", timeout=6000)
def generate_images(prompt, input_image_b64=None, strength=0.7, guidance_scale=8.0, steps=30):
    results = {}
    input_image = decode_img(input_image_b64) if input_image_b64 else None

    try:
        vae = AutoencoderKL.from_pretrained(MODELS["vae"], torch_dtype=torch.float16)

        # SDXL base + refiner
        base_pipe = StableDiffusionXLPipeline.from_pretrained(
            MODELS["base"],
            torch_dtype=torch.float16,
            vae=vae,
            use_safetensors=True,
            variant="fp16"
        )
        base_pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            base_pipe.scheduler.config,
            use_karras_sigmas=True,
            algorithm_type="sde-dpmsolver++"
        )
        base_pipe.enable_model_cpu_offload()

        refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            MODELS["refiner"],
            torch_dtype=torch.float16,
            vae=vae
        )
        refiner.enable_model_cpu_offload()

        generator = torch.manual_seed(1)
        enhanced_prompt = f"{prompt}, masterpiece, 4K, best quality"

        if input_image and strength > 0:
            latent = base_pipe(
                prompt=enhanced_prompt,
                image=input_image,
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
                output_type="latent"
            ).images[0]
        else:
            latent = base_pipe(
                prompt=enhanced_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
                width=1024,
                height=1024,
                output_type="latent"
            ).images[0]

        refined = refiner(
            prompt=enhanced_prompt,
            image=latent,
            num_inference_steps=max(10, steps // 2),
            generator=generator
        ).images[0]

        results["base"] = encode_img(refined)

        if input_image:
            # Depth
            try:
                depth_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas")
                depth_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")
                depth_inputs = depth_extractor(images=input_image, return_tensors="pt")
                with torch.no_grad():
                    depth = depth_model(**depth_inputs).predicted_depth
                depth = torch.nn.functional.interpolate(depth.unsqueeze(1), size=(1024, 1024), mode="bicubic").squeeze().cpu().numpy()
                depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255
                depth_img = Image.fromarray(depth.astype("uint8")).convert("RGB")
                results["depth"] = get_controlnet_output(prompt, depth_img, MODELS["depth"], vae, seed=2)
            except Exception as e:
                print(f"Depth failed: {str(e)}")
                results["depth"] = "generation_failed"

            # Pose
            try:
                pose_detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
                pose_img = pose_detector(input_image)
                results["pose"] = get_controlnet_output(prompt, pose_img, MODELS["pose"], vae, seed=3)
            except Exception as e:
                print(f"Pose failed: {str(e)}")
                results["pose"] = "generation_failed"

            # Canny
            try:
                canny_edges = cv2.Canny(np.array(input_image), 100, 200)
                canny_img = Image.fromarray(canny_edges).convert("RGB")
                results["canny"] = get_controlnet_output(prompt, canny_img, MODELS["canny"], vae, seed=4)
            except Exception as e:
                print(f"Canny failed: {str(e)}")
                results["canny"] = "generation_failed"

    except Exception as e:
        results["error"] = f"Generation failed: {str(e)}"

    return results

@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(title="Multi-Model Image API")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    class RequestModel(BaseModel):
        prompt: str
        input_image_b64: str = None
        strength: float = 0.7
        guidance_scale: float = 8.0
        steps: int = 30

    @app.post("/compare")
    async def generate(request: RequestModel):
        try:
            # Using synchronous remote call (no await needed)
            result = generate_images.remote(
                prompt=request.prompt,
                input_image_b64=request.input_image_b64,
                strength=request.strength,
                guidance_scale=request.guidance_scale,
                steps=request.steps
            )
            if "error" in result:
                raise HTTPException(500, result["error"])
            return result
        except Exception as e:
            raise HTTPException(500, detail=str(e))

    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "models": list(MODELS.keys())}

    return app