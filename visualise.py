import cv2
import os
import numpy as np

def draw_bounding_box(image, bbox, color=(0, 255, 0), thickness=2):
    """Draw a bounding box on the image."""
    h, w = image.shape[:2]
    x_center, y_center, width, height = bbox
    x1 = int((x_center - width/2) * w)
    y1 = int((y_center - height/2) * h)
    x2 = int((x_center + width/2) * w)
    y2 = int((y_center + height/2) * h)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

def visualize_bounding_boxes(image_path, label_path):
    """Visualize bounding boxes on an image."""
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image at {image_path}")
        return

    # Read the label file
    with open(label_path, 'r') as file:
        lines = file.readlines()

    # Draw bounding boxes for each object
    for line in lines:
        class_id, x_center, y_center, width, height = map(float, line.strip().split())
        draw_bounding_box(image, (x_center, y_center, width, height))

    # Display the image
    # cv2.imshow('Image with Bounding Boxes', image)
    cv2.imwrite('visual_image/output_image.jpg', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# Directory paths
image_dir = '/home/rbhalekar/finally_merged_dataset1/train/images/00001_80.jpg' 
label_dir = '/home/rbhalekar/finally_merged_dataset1/train/labels/00001_80.txt'



visualize_bounding_boxes(image_dir,label_dir)
