from diffusers import StableDiffusionPipeline
import torch

# Ensure you have the right version of diffusers
# !pip install diffusers==0.28.0

# Load the pipeline
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id)

# Move the pipeline to multiple GPUs
pipe = pipe.to("cuda")

# Specify multiple GPUs
device_ids = [0, 1]  # Example: using GPU 0 and 1
pipe = torch.nn.DataParallel(pipe, device_ids=device_ids)

# Perform inference
prompt = "A beautiful landscape with mountains and rivers"
generator = torch.Generator(device="cuda").manual_seed(42)  # For reproducibility

# Generate image
with torch.no_grad():
    images = pipe(prompt, generator=generator)["sample"]

# Save the generated image
images[0].save("output.png")
