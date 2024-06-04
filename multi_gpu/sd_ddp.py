from diffusers import StableDiffusionXLPipeline
import torch
from torch.nn import DataParallel

# Ensure you have the right version of diffusers
#!pip install diffusers==0.28.0 transformers

# Load the pipeline with the model
pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl",
    torch_dtype=torch.float16
)

# Move the pipeline to multiple GPUs using DataParallel
device_ids = list(range(torch.cuda.device_count()))  # Get a list of all available GPUs
pipeline = DataParallel(pipeline, device_ids=device_ids)
pipeline = pipeline.to(f'cuda:{pipeline.device_ids[0]}')

# List of prompts for testing
prompts = [
    "a futuristic cityscape",
    "a serene beach at sunset",
    "a bustling market in a medieval town",
    "a space station orbiting Earth",
    "a majestic mountain range with a clear blue sky",
    "a dense forest with sunlight filtering through the trees",
    "a robot exploring an alien planet",
    "a fantasy castle surrounded by dragons",
    "a cyberpunk city at night with neon lights",
    "an underwater scene with colorful coral reefs",
    # Add more prompts as needed
]

# Function to generate and save images
def generate_images(prompts):
    for i, prompt in enumerate(prompts):
        # Perform inference
        image = pipeline(prompt).images[0]
        # Save the generated image
        image.save(f"output_{i}.png")
        print(f"Generated and saved image for prompt: {prompt}")

# Generate images for all prompts
generate_images(prompts)
