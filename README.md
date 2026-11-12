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




Aim
To write and execute a Python program to implement a generative model for image editing using a pre-trained Stable Diffusion Inpainting model.
2. Procedure
1. Install the required Python libraries.
2. Import the necessary libraries such as PyTorch, Diffusers and PIL.
3. Load the input image and mask image.
4. Load the pre-trained Stable Diffusion Inpainting model.
5. Give a text prompt describing the required image modification.
6. Generate the edited image using the generative model.
7. Display and save the generated image.
8. Verify the output.
3. Required Installation
Open the VS Code terminal and execute:
