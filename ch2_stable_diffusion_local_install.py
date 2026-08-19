import torch
from diffusers import StableDiffusionPipeline

# Load the lightweight model
print("Loading model locally... (this may take a minute on first run)")
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
)

# Move the model to your Mac's Apple Silicon GPU (MPS)
pipe = pipe.to("mps")

# Optional: Enable memory-efficient optimizations for Mac
pipe.enable_attention_slicing()

print("Generating image...")
prompt = "A detailed technical diagram of an AI agent"
image = pipe(prompt).images[0]

# Save the image locally
image.save("local_ai_agent.png")
print("Image saved successfully as local_ai_agent.png!")
