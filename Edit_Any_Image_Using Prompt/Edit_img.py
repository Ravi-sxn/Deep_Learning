
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from PIL import Image, ImageDraw, ImageFont

# Load the image from the file path

image_path = "Input_img/img_2.jpg"
image = Image.open(image_path)

model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
pipe.to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

print("Enter prompt for editing the Image")
prompt=str(input())



# Generate the edited image
edited_image = pipe(prompt, image=image).images[0]

# Create a drawing context on the edited image
draw = ImageDraw.Draw(edited_image)

# Set the font and size (you may need to adjust the path to a font file or remove if default is fine)
try:
    font = ImageFont.truetype("arial.ttf", 20)  # Adjust the font size as needed
except IOError:
    font = ImageFont.load_default()  # Fallback to default font if arial.ttf is not available

# Define text position (bottom-left corner with some padding)
text_position = (10, edited_image.height - 30)

# Add the prompt as text on the edited image
draw.text(text_position, prompt, font=font, fill="white")

# Save the edited image with the prompt written on it
output_path = 'Output/Edited_img/edit_img_2.png'
edited_image.save(output_path)


