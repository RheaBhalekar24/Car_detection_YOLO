from ultralytics import YOLO

import os
import shutil
import yaml


#define the model
model = YOLO('/home/rbhalekar/ml_project_part1/yolo11l.pt')

yaml_path = '/home/rbhalekar/ml_project_part1/final_yaml.yaml'

model.train(data= yaml_path, epochs=100, imgsz=640)
model.save('/home/rbhalekar/ml_project_part1/trained_yolo11l.pt')

