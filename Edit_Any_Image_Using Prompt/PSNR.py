import cv2
import torch
import numpy as np
import lpips
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import torchvision.transforms as transforms


# Load LPIPS model (perceptual similarity metric)
lpips_model = lpips.LPIPS(net='alex')  # Can use 'vgg' or 'alex'

# Preprocess images: function to load and resize images
def preprocess_image(image_path, target_size=(256,256)):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor()
    ])
    img = transform(img).unsqueeze(0)  # Add batch dimension
    return img

# Convert tensor to numpy array for PSNR and SSIM
def tensor_to_numpy(tensor_img):
    img = tensor_img.squeeze(0).permute(1, 2, 0).cpu().numpy()  # Convert to HWC
    img = (img * 255).astype(np.uint8)  # Convert to uint8
    return img




# PSNR and SSIM calculation
def calculate_psnr_ssim(img1, img2):
    img1_np = tensor_to_numpy(img1)
    img2_np = tensor_to_numpy(img2)
    
    # Ensure the window size for SSIM fits the image dimensions
    min_size = min(img1_np.shape[0], img1_np.shape[1])  # Find the smaller of height or width
    win_size = min(7, min_size)  # Use a window size of 7 or smaller if the image is too small
    
    psnr_value = psnr(img1_np, img2_np)
    
    # Use channel_axis for multichannel (color) images
    ssim_value = ssim(img1_np, img2_np, win_size=win_size, channel_axis=2)  # channel_axis=2 for RGB
    
    return psnr_value, ssim_value



# LPIPS calculation
def calculate_lpips(img1, img2):
    lpips_value = lpips_model(img1, img2)
    return lpips_value.item()

# Paths to your images
original_image_path = 'Input_img/img_2.jpg'
edited_image_1_path = 'Output/Edited_img/edit_img_2.png'
edited_image_2_path = 'Output/Gen_SD3_img/SD3_img_2.png'



# Load and preprocess the images
original_image = preprocess_image(original_image_path)
edited_image_1 = preprocess_image(edited_image_1_path)
edited_image_2 = preprocess_image(edited_image_2_path)

# PSNR and SSIM between Original and Edited Image 1
psnr_value_1, ssim_value_1 = calculate_psnr_ssim(original_image, edited_image_1)
print(f'PSNR between Original and Edited Image 1: {psnr_value_1:.4f}')
print(f'SSIM between Original and Edited Image 1: {ssim_value_1:.4f}')

# PSNR and SSIM between Original and Edited Image 2
psnr_value_2, ssim_value_2 = calculate_psnr_ssim(original_image, edited_image_2)
print(f'PSNR between Original and Edited Image 2: {psnr_value_2:.4f}')
print(f'SSIM between Original and Edited Image 2: {ssim_value_2:.4f}')

# LPIPS between Original and Edited Image 1
lpips_value_1 = calculate_lpips(original_image, edited_image_1)
print(f'LPIPS between Original and Edited Image 1: {lpips_value_1:.4f}')

# LPIPS between Original and Edited Image 2
lpips_value_2 = calculate_lpips(original_image, edited_image_2)
print(f'LPIPS between Original and Edited Image 2: {lpips_value_2:.4f}')
