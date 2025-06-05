import modal

app = modal.App("multi-model-image-api")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch", "diffusers", "transformers", "accelerate", "opencv-python",
        "safetensors", "Pillow", "fastapi", "pydantic", "mediapipe", 
        "controlnet-aux>=0.0.6"  # Required for DWOpenPose
    )
)

@app.function(
    image=image,
    gpu="A10G",
    timeout=600,
    min_containers=1
)
def generate_images(prompt: str, input_image_b64: str, strength: float = 0.6, guidance_scale: float = 7.5, steps: int = 20) -> dict:
    import torch
    from diffusers import (
        StableDiffusionControlNetPipeline,
        ControlNetModel,
        UniPCMultistepScheduler,
    )
    from PIL import Image
    import numpy as np
    import base64
    import io
    import cv2
    from transformers import DPTFeatureExtractor, DPTForDepthEstimation
    from controlnet_aux import OpenposeDetector, CannyDetector, NormalBaeDetector, DWPoseDetector

    def decode_img(b64):
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB").resize((768, 768))

    def encode_img(img):
        buffer = io.BytesIO()
        img.save(buffer, format="WEBP")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    input_image = decode_img(input_image_b64)
    images_out = {}

    # === SLOT 1: ControlNet (Canny)
    canny_np = cv2.Canny(np.array(input_image), 100, 200)
    canny_pil = Image.fromarray(cv2.cvtColor(canny_np, cv2.COLOR_GRAY2RGB))
    controlnet_canny = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
    pipe_canny = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet_canny, torch_dtype=torch.float16)
    pipe_canny.scheduler = UniPCMultistepScheduler.from_config(pipe_canny.scheduler.config)
    pipe_canny.enable_model_cpu_offload()
    images_out["controlnet_canny"] = encode_img(pipe_canny(
        prompt=prompt + ", edge-enhanced, clean lines",
        image=canny_pil,
        num_inference_steps=steps,
        generator=torch.manual_seed(1),
        guidance_scale=guidance_scale,
        negative_prompt="blurry, low resolution"
    ).images[0])

    # === SLOT 2: ControlNet (Depth)
    depth_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas")
    depth_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")
    depth_inputs = depth_extractor(images=input_image, return_tensors="pt")
    with torch.no_grad():
        depth_output = depth_model(**depth_inputs)
        depth = depth_output.predicted_depth
    depth = torch.nn.functional.interpolate(depth.unsqueeze(1), size=(768, 768), mode="bicubic", align_corners=False)
    depth = depth.squeeze().cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255
    depth_pil = Image.fromarray(depth.astype("uint8")).convert("RGB")
    controlnet_depth = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16)
    pipe_depth = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet_depth, torch_dtype=torch.float16)
    pipe_depth.scheduler = UniPCMultistepScheduler.from_config(pipe_depth.scheduler.config)
    pipe_depth.enable_model_cpu_offload()
    images_out["controlnet_depth"] = encode_img(pipe_depth(
        prompt=prompt + ", 3D lighting, realistic depth",
        image=depth_pil,
        num_inference_steps=steps,
        generator=torch.manual_seed(2),
        guidance_scale=guidance_scale,
        negative_prompt="blurry, low resolution"
    ).images[0])

    # === SLOT 3: ControlNet (OpenPose)
    pose_detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
    pose_img = pose_detector(input_image)
    controlnet_pose = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16)
    pipe_pose = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet_pose, torch_dtype=torch.float16)
    pipe_pose.scheduler = UniPCMultistepScheduler.from_config(pipe_pose.scheduler.config)
    pipe_pose.enable_model_cpu_offload()
    images_out["controlnet_pose"] = encode_img(pipe_pose(
        prompt=prompt + ", full body, dynamic pose",
        image=pose_img,
        num_inference_steps=steps,
        generator=torch.manual_seed(3),
        guidance_scale=guidance_scale,
        negative_prompt="blurry, distorted"
    ).images[0])

    # === SLOT 4: ControlNet (Scribble)
    scribble_detector = CannyDetector()
    scribble_img = scribble_detector(input_image)
    controlnet_scribble = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-scribble", torch_dtype=torch.float16)
    pipe_scribble = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet_scribble, torch_dtype=torch.float16)
    pipe_scribble.scheduler = UniPCMultistepScheduler.from_config(pipe_scribble.scheduler.config)
    pipe_scribble.enable_model_cpu_offload()
    images_out["controlnet_scribble"] = encode_img(pipe_scribble(
        prompt=prompt + ", sketch style, clean outline",
        image=scribble_img,
        num_inference_steps=steps,
        generator=torch.manual_seed(4),
        guidance_scale=guidance_scale,
        negative_prompt="incomplete, noisy"
    ).images[0])

    # === SLOT 5: ControlNet (Normal Map)
    normal_model_id = "lllyasviel/Annotators"
    normal_detector = NormalBaeDetector.from_pretrained(normal_model_id)
    normal_img = normal_detector(input_image)
    controlnet_normal = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-normal", torch_dtype=torch.float16)
    pipe_normal = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet_normal, torch_dtype=torch.float16)
    pipe_normal.scheduler = UniPCMultistepScheduler.from_config(pipe_normal.scheduler.config)
    pipe_normal.enable_model_cpu_offload()
    images_out["controlnet_normal"] = encode_img(pipe_normal(
        prompt=prompt + ", detailed shading, smooth surfaces",
        image=normal_img,
        num_inference_steps=steps,
        generator=torch.manual_seed(5),
        guidance_scale=guidance_scale,
        negative_prompt="flat, noisy"
    ).images[0])

    # === SLOT 6: DWOpenPose (DensePose + OpenPose) - REPLACED COMBINED CANNY+OPENPOSE
    dwpose_detector = DWPoseDetector.from_pretrained("yzd-v/DWPose")
    dwpose_img = dwpose_detector(input_image)
    controlnet_dwpose = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16)
    pipe_dwpose = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet_dwpose,
        torch_dtype=torch.float16
    )
    pipe_dwpose.scheduler = UniPCMultistepScheduler.from_config(pipe_dwpose.scheduler.config)
    pipe_dwpose.enable_model_cpu_offload()
    images_out["dwpose"] = encode_img(pipe_dwpose(
        prompt=prompt + ", anatomical accuracy, dynamic pose",
        image=dwpose_img,
        num_inference_steps=steps,
        generator=torch.manual_seed(6),
        guidance_scale=guidance_scale,
        negative_prompt="blurry, distorted anatomy"
    ).images[0])

    return images_out

@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    from fastapi.middleware.cors import CORSMiddleware
    import traceback

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
        strength: float = 0.6
        guidance_scale: float = 7.5
        steps: int = 20

    @app.post("/compare")
    async def compare(request: RequestModel):
        try:
            images = generate_images.remote(
                prompt=request.prompt,
                input_image_b64=request.input_image_b64,
                strength=request.strength,
                guidance_scale=request.guidance_scale,
                steps=request.steps
            )
            return {
                "images": images,
                "models_used": list(images.keys())
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=traceback.format_exc())

    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "models": [
                "controlnet_canny",
                "controlnet_depth",
                "controlnet_pose",
                "controlnet_scribble",
                "controlnet_normal",
                "dwpose"  # Updated from combined_canny_pose
            ]
        }

    return app