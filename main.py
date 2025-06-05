import modal

app = modal.App("multi-model-image-api")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch", "diffusers", "transformers", "accelerate", "opencv-python",
        "safetensors", "Pillow", "fastapi", "pydantic"
    )
)

@app.function(
    image=image,
    gpu="A10G",
    timeout=600,
    min_containers=1
)
def generate_images(prompt: str, input_image_b64: str) -> dict:
    import torch
    from diffusers import (
        StableDiffusionControlNetPipeline,
        ControlNetModel,
        UniPCMultistepScheduler,
        StableDiffusionXLPipeline,
    )
    from PIL import Image
    import numpy as np
    import base64
    import io
    import cv2
    from transformers import DPTFeatureExtractor, DPTForDepthEstimation

    def decode_img(b64):
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB").resize((512, 512))

    def encode_img(img):
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    input_image = decode_img(input_image_b64)

    # === SLOT 1: ControlNet (Canny)
    canny_np = cv2.Canny(np.array(input_image), 100, 200)
    canny_pil = Image.fromarray(cv2.cvtColor(canny_np, cv2.COLOR_GRAY2RGB))

    controlnet_canny = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16
    )
    pipe_canny = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet_canny,
        torch_dtype=torch.float16,
    )
    pipe_canny.scheduler = UniPCMultistepScheduler.from_config(pipe_canny.scheduler.config)
    pipe_canny.enable_model_cpu_offload()

    result_canny = pipe_canny(
        prompt=prompt,
        image=canny_pil,
        num_inference_steps=20,
        generator=torch.manual_seed(1),
        negative_prompt="blurry, low resolution"
    ).images[0]

    # === SLOT 2: ControlNet (Depth)
    depth_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas")
    depth_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")

    depth_inputs = depth_extractor(images=input_image, return_tensors="pt")
    with torch.no_grad():
        depth_output = depth_model(**depth_inputs)
        depth = depth_output.predicted_depth
    depth = torch.nn.functional.interpolate(
        depth.unsqueeze(1),
        size=(512, 512),
        mode="bicubic",
        align_corners=False,
    )
    depth = depth.squeeze().cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255
    depth_pil = Image.fromarray(depth.astype("uint8")).convert("RGB")

    controlnet_depth = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16
    )
    pipe_depth = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet_depth,
        torch_dtype=torch.float16,
    )
    pipe_depth.scheduler = UniPCMultistepScheduler.from_config(pipe_depth.scheduler.config)
    pipe_depth.enable_model_cpu_offload()

    result_depth = pipe_depth(
        prompt=prompt,
        image=depth_pil,
        num_inference_steps=20,
        generator=torch.manual_seed(2),
        negative_prompt="blurry, low resolution"
    ).images[0]

    # === SLOT 3: SDXL Turbo (img2img)
    sdxl_img2img = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
    ).to("cuda")
    sdxl_img2img.enable_model_cpu_offload()

    result_sdxl = sdxl_img2img(
        prompt=prompt,
        image=input_image,
        num_inference_steps=4,
        guidance_scale=0.0,
        strength=0.6
    ).images[0]

    return {
        "controlnet_canny": encode_img(result_canny),
        "controlnet_depth": encode_img(result_depth),
        "sdxl_img2img": encode_img(result_sdxl)
    }

@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(title="Multi-Model Image Generator")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    class RequestModel(BaseModel):
        prompt: str
        input_image_b64: str

    @app.post("/compare")
    async def compare(request: RequestModel):
        try:
            images = generate_images.remote(request.prompt, request.input_image_b64)
            return {
                "images": images,
                "models_used": ["controlnet_canny", "controlnet_depth", "sdxl_img2img"]
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "models": ["controlnet_canny", "controlnet_depth", "sdxl_img2img"]
        }

    return app
