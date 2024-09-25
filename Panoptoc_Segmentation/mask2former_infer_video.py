from PIL import Image
from model import load_model
from color_palette import ADE_CATEGORIES
from utils import draw_segmentation_map, image_overlay

import argparse
import torch
import cv2
import time
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    '--input',
    default='input/video_1.mp4'    
)
args = parser.parse_args()

out_dir = 'outputs'
os.makedirs(out_dir, exist_ok=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model, processor = load_model('semantic')
model = model.to(device).eval()

cap = cv2.VideoCapture(args.input)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
vid_fps = int(cap.get(5))
save_name = args.input.split(os.path.sep)[-1].split('.')[0]

# Define codec and create VideoWriter object.
out = cv2.VideoWriter(
    f"{out_dir}/{save_name}.mp4", 
    cv2.VideoWriter_fourcc(*'mp4v'), 
    vid_fps, 
    (frame_width, frame_height)
)

frame_count = 0 # To count total frames.
total_fps = 0 # To get the final frames per second.

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        frame_count += 1
        # frame = cv2.resize(frame, resize)
        image = Image.fromarray(frame).convert('RGB')
        inputs = processor(image, return_tensors='pt').to(device)
        
        start_time = time.time()

        with torch.no_grad():
            outputs = model(**inputs)
        
        # Model predicts class_queries_logits of shape `(batch_size, num_queries)`
        # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
        class_queries_logits = outputs.class_queries_logits
        masks_queries_logits = outputs.masks_queries_logits

        pred_map = processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]]
        )[0]

        mask = pred_map

        mask = mask.cpu()
        seg_map = draw_segmentation_map(mask, ADE_CATEGORIES)
        result = image_overlay(image, seg_map)

        end_time = time.time()

        fps = 1 / (end_time - start_time)
        total_fps += fps

        cv2.putText(
            result, 
            f"FPS: {fps:.1f}", 
            (15, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale=1, 
            color=(0, 0, 255), 
            thickness=2, 
            lineType=cv2.LINE_AA
        )
        
        end_time = time.time()

        out.write(result)

        cv2.imshow('Image', result)
        # Press `q` to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

# Release VideoCapture().
cap.release()
# Close all frames and video windows.
cv2.destroyAllWindows()

# Calculate and print the average FPS.
avg_fps = total_fps / frame_count
print(f"Average FPS: {avg_fps:.3f}")