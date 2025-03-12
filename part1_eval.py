from ultralytics import YOLO

import os
import shutil
import yaml

model = YOLO('/home/rbhalekar/finally_merged_dataset1/trained_yolo11l.pt')
yaml_path = '/home/rbhalekar/finally_merged_dataset1/final_yaml.yaml'

metrics = model.val(data= yaml_path)
print(metrics.box.map)  # map50-95

