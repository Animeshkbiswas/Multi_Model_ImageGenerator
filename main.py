import modal
import os
HF_TOKEN = os.environ.get("HF_TOKEN")
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLInstructPix2PixPipeline,
    DPMSolverMultistepScheduler,
    AutoencoderKL,
    ControlNetModel,
    EDMEulerScheduler,
)
from PIL import Image
import numpy as np
import base64
import io
import torch
from transformers import DPTFeatureExtractor, DPTForDepthEstimation
from controlnet_aux.open_pose import OpenposeDetector
from huggingface_hub import hf_hub_download, InferenceClient
import cv2

app = modal.App("multi-model-image-api")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgl1", "libglib2.0-0", "libsm6", "libxext6")
    .pip_install(
        "numpy<2.0",
        "torch==2.1.2",
        "diffusers==0.27.2",
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
    "pose": "thibaud/controlnet-openpose-sdxl-1.0",
    "canny": "diffusers/controlnet-canny-sdxl-1.0",
    "vae": "madebyollin/sdxl-vae-fp16-fix"
}
@app.function(secrets=[modal.Secret.from_name("huggingface-secret")])
def f():
   HF_TOKEN = print(os.environ["HF_TOKEN"])
# Prompt Enhancer using Zephyr LLM
client1 = InferenceClient(
    model="HuggingFaceH4/zephyr-7b-beta",
    token= HF_TOKEN
)
system_instructions1 = """<|system|>
Act as an Image Prompt Generation expert. Your task is to rewrite and enhance USER's prompt for image generation using Stable Diffusion XL.
Ensure the enhanced prompt includes key descriptors for:
- quality (e.g., 4K, masterpiece, ultra-detailed)
- style (e.g., realistic)
- lighting/composition (e.g., cinematic lighting)

Keep it concise but expressive. Add style/quality keywords at the end.
<|user|>
"""

def promptifier(prompt: str) -> str:
    formatted_prompt = f"{system_instructions1}{prompt}\n<|assistant|>\n"
    try:
        enhanced = client1.text_generation(formatted_prompt, max_new_tokens=100)
        return enhanced.strip()
    except Exception as e:
        print(f"Prompt enhancement failed: {e}")
        return prompt  # Fallback

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

def get_controlnet_output(prompt, guide_img, model_id, vae, refiner=None, seed=1, refine_strength=0.3):
    try:
        controlnet = ControlNetModel.from_pretrained(model_id, torch_dtype=torch.float16)
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            MODELS["base"],
            controlnet=controlnet,
            torch_dtype=torch.float16,
            vae=vae
        )
        pipe.enable_model_cpu_offload()

        image = pipe(
            prompt=prompt,
            image=guide_img,
            num_inference_steps=30,
            guidance_scale=8.0,
            generator=torch.manual_seed(seed)
        ).images[0]

        if refiner and refine_strength > 0:
            image = refiner(
                prompt=prompt,
                image=image,
                strength=refine_strength,
                num_inference_steps=max(10, int(30 * refine_strength)),
                guidance_scale=5.0,
                generator=torch.manual_seed(seed)
            ).images[0]

        return encode_img(image)
    except Exception as e:
        print(f"ControlNet failed for {model_id}: {str(e)}")
        return "generation_failed"

@app.function(image=image, gpu="A100-40GB", secrets=[modal.Secret.from_name("huggingface-secret")], timeout=6000)
def generate_images(prompt, input_image_b64=None, strength=0.7, guidance_scale=8.0, steps=30):
    results = {}
    input_image = decode_img(input_image_b64) if input_image_b64 else None

    try:
        vae = AutoencoderKL.from_pretrained(MODELS["vae"], torch_dtype=torch.float16)

        refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            MODELS["refiner"],
            torch_dtype=torch.float16,
            vae=vae
        )
        refiner.enable_model_cpu_offload()

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

        # Apply prompt enhancement
        enhanced_prompt = promptifier(prompt)
        print(f"Enhanced prompt: {enhanced_prompt}")

        generator = torch.manual_seed(1)

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
            try:
                depth_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas")
                depth_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")
                depth_inputs = depth_extractor(images=input_image, return_tensors="pt")
                with torch.no_grad():
                    depth = depth_model(**depth_inputs).predicted_depth
                depth = torch.nn.functional.interpolate(
                    depth.unsqueeze(1),
                    size=(1024, 1024),
                    mode="bicubic"
                ).squeeze().cpu().numpy()
                depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255
                depth_img = Image.fromarray(depth.astype("uint8")).convert("RGB")
                results["depth"] = get_controlnet_output(
                    enhanced_prompt, depth_img, MODELS["depth"], vae, refiner, seed=2, refine_strength=0.3
                )
            except Exception as e:
                print(f"Depth failed: {str(e)}")
                results["depth"] = "generation_failed"

            try:
                pose_detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
                pose_img = pose_detector(input_image)
                results["pose"] = get_controlnet_output(
                    enhanced_prompt, pose_img, MODELS["pose"], vae, refiner, seed=3, refine_strength=0.2
                )
            except Exception as e:
                print(f"Pose failed: {str(e)}")
                results["pose"] = "generation_failed"

            try:
                canny_edges = cv2.Canny(np.array(input_image), 100, 200)
                canny_img = Image.fromarray(canny_edges).convert("RGB")
                results["canny"] = get_controlnet_output(
                    enhanced_prompt, canny_img, MODELS["canny"], vae, refiner, seed=4, refine_strength=0.4
                )
            except Exception as e:
                print(f"Canny failed: {str(e)}")
                results["canny"] = "generation_failed"

            try:
                cosxl_edit_path = hf_hub_download(
                 repo_id="stabilityai/cosxl",
                  filename="cosxl_edit.safetensors",
                  token=HF_TOKEN
                )

                edit_pipe = StableDiffusionXLInstructPix2PixPipeline.from_single_file(
                    cosxl_edit_path,
                    num_in_channels=8,
                    is_cosxl_edit=True,
                    vae=vae,
                    torch_dtype=torch.float16
                )
                edit_pipe.scheduler = EDMEulerScheduler(
                    sigma_min=0.002,
                    sigma_max=120.0,
                    sigma_data=1.0,
                    prediction_type="v_prediction"
                )
                edit_pipe.to("cuda")

                edited = edit_pipe(
                    prompt=enhanced_prompt,
                    image=input_image,
                    guidance_scale=7.5,
                    num_inference_steps=30,
                    generator=torch.manual_seed(5)
                ).images[0]

                results["edit"] = encode_img(edited)
            except Exception as e:
                print(f"InstructPix2Pix failed: {str(e)}")
                results["edit"] = "generation_failed"

    except Exception as e:
        results["error"] = f"Generation failed: {str(e)}"

    return results

@app.function(image=image, timeout=6000, secrets=[modal.Secret.from_name("huggingface-secret")])
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
