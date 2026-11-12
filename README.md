
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image

# Load model
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stable-diffusion-2-inpainting"
)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# Load images
image = Image.open("input.jpg").convert("RGB").resize((512, 512))
mask = Image.open("mask.png").convert("RGB").resize((512, 512))

# Generate edited image
result = pipe(
    prompt="A beautiful garden with flowers",
    image=image,
    mask_image=mask
).images[0]

result.save("edited.png")
print("Image edited successfully")

