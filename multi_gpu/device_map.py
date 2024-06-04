from diffusers import StableDiffusionXLPipeline
import torch

# Ensure you have the right version of diffusers
!pip install diffusers==0.28.0

# Load the pipeline with device_map for SDXL
pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", 
    torch_dtype=torch.float16, 
    device_map="balanced"
)

# Perform inference
prompt = "a futuristic cityscape"
image = pipeline(prompt).images[0]
