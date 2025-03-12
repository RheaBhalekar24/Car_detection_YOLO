from ultralytics import YOLO
import cv2
import numpy as np
import random
import os
import yaml

# Load your trained model
model = YOLO('/home/rbhalekar/ml_project_part1/trained_yolo11l.pt')

# Load your dataset configuration
with open('/home/rbhalekar/finally_merged_dataset1/final_yaml.yaml', 'r') as file:
    data_config = yaml.safe_load(file)

# Path to validation images
val_images_path = data_config['val']

# Get a list of all image files
image_files = [f for f in os.listdir(val_images_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Randomly select 10 images
selected_images = random.sample(image_files, 10)

def draw_boxes(img, boxes, color):
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    return img

for img_file in selected_images:
    img_path = os.path.join(val_images_path, img_file)
    
    # Read the image
    img = cv2.imread(img_path)
    
    # Get ground truth boxes
    label_path = img_path.replace('images', 'labels').rsplit('.', 1)[0] + '.txt'
    gt_boxes = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                cls, x, y, w, h = map(float, line.strip().split())
                x1 = int((x - w/2) * img.shape[1])
                y1 = int((y - h/2) * img.shape[0])
                x2 = int((x + w/2) * img.shape[1])
                y2 = int((y + h/2) * img.shape[0])
                gt_boxes.append([x1, y1, x2, y2])
    
    # Get predicted boxes
    results = model(img)
    pred_boxes = results[0].boxes.xyxy.cpu().numpy()
    
    # Draw ground truth boxes in green
    img = draw_boxes(img, gt_boxes, (0, 255, 0))
    
    # Draw predicted boxes in blue
    img = draw_boxes(img, pred_boxes, (255, 0, 0))

    output_image_dir = '/home/rbhalekar/ml_project_part1/output_images'
    os.makedirs(output_image_dir, exist_ok=True)

    output_path = os.path.join(output_image_dir, f'visualization_{img_file}')
    cv2.imwrite(output_path, img)
    print(f"Saved visualization for {img_file} to {output_path}")
    


print("Visualization complete.")
