from PIL import Image
from model import load_model
from color_palette import (
    COCO_INSTANCE_CATEGORIES, 
    COCO_PANOPTIC_CATEGORIES, 
    ADE_CATEGORIES
)
from utils import draw_segmentation_map, image_overlay

import argparse
import torch
import cv2
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    '--input',
    default='input/image_1.jpg'    
)
parser.add_argument(
    '--task',
    default='semantic'
)
args = parser.parse_args()

out_dir = 'outputs'
os.makedirs(out_dir, exist_ok=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model, processor = load_model(args.task)
model = model.to(device).eval()

image = Image.open(args.input)
inputs = processor(image, return_tensors='pt').to(device)

with torch.no_grad():
    outputs = model(**inputs)

if args.task == 'semantic':
    pred_map = processor.post_process_semantic_segmentation(
        outputs, target_sizes=[image.size[::-1]]
    )[0]
    mask = pred_map.cpu()
    categories = ADE_CATEGORIES

if args.task == 'instance':
    pred_map = processor.post_process_instance_segmentation(
        outputs, target_sizes=[image.size[::-1]]
    )[0]
    mask = pred_map['segmentation'].cpu()
    categories = COCO_INSTANCE_CATEGORIES

if args.task == 'panoptic':
    pred_map = processor.post_process_panoptic_segmentation(
        outputs, target_sizes=[image.size[::-1]]
    )[0]
    mask = pred_map['segmentation'].cpu()
    categories = COCO_PANOPTIC_CATEGORIES

seg_map = draw_segmentation_map(mask, categories)
result = image_overlay(image, seg_map)

save_name = args.task + '_' + args.input.split(os.path.sep)[-1]
cv2.imwrite(os.path.join(out_dir, save_name), result)
cv2.imshow('Image', result)
cv2.waitKey(0)