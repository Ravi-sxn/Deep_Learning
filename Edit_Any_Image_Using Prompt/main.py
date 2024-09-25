import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image
from Edit_img import prompt
from PIL import Image, ImageDraw, ImageFont

pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipe = pipe.to("cuda")


init_image = Image.open("Input_img/img_2.jpg")
prompt_2 = prompt
image = pipe(prompt_2, image=init_image).images[0]
image.save("Output/Gen_SD3_img/SD3_img_2.png")
