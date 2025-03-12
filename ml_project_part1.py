
"""Ml_project_part1.ipynb

The file is for creating the Dataloader and converting the 
bounding boxes from 3D to 2D

"""

import pdb
import json
import zipfile
import os
import yaml
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.spatial.transform import Rotation as R
from matplotlib.patches import Rectangle
from torch.utils.data import random_split

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

'''
this is your dataset directory 
const_output_dir - your main directory path 
output_dir20m, output_dir50m output_dir80m - path to the image directory 
'''
const_output_dir = '/home/rbhalekar/Town01_Opt_120/ClearNoon'
output_dir20m = '/home/rbhalekar/Town01_Opt_120/ClearNoon/height20m'
output_dir50m = '/home/rbhalekar/Town01_Opt_120/ClearNoon/height50m'
output_dir80m = '/home/rbhalekar/Town01_Opt_120/ClearNoon/height80m'

print(os.listdir(const_output_dir))
# print(os.listdir(output_dir))


print((os.getcwd()))
from ultralytics import YOLO

#camera matrix 

K = np.array([
            [  0, 960, 960],
            [960,   0, 540],
            [  0,   0,   1]
        ])

#identifiers for car from Syndrone Dataset
idmap = {(100,):1}
car_identifiers = { 'id': 100 , 'cmap' :  [  0,  0,142]}

#function for converting 3D bounding boxes into 2D bounding boxes 

def bounding_boxes(idx, output_dir,iden = car_identifiers, camera_matrix = K):
  # opening rgb bbox and camera extrinsic
  rgb_image = cv.imread(f'{output_dir}/rgb/{idx:05d}.jpg',cv.IMREAD_UNCHANGED)[...,::-1]/255.
  sem = cv.imread(f'{output_dir}/semantic/{idx:05d}.png')
  camera_extrinsic = json.load(open(f'{output_dir}/camera/{idx:05d}.json'))

  shift = np.array([camera_extrinsic['x'], camera_extrinsic['y'], camera_extrinsic['z']])
  # creates a rotation matrix to transform
  rotation = R.from_euler('yzx', [90-camera_extrinsic['pitch'], camera_extrinsic['yaw'], camera_extrinsic['roll']], degrees=True).as_matrix()

  # opening the bounding box
  # bbpath = f'{const_output_dir}/bboxes/{idx:05d}.json'
  bbpath = f'{const_output_dir}/bboxes/{idx:05d}.json'

  bboxes = json.load(open(bbpath))

  # transform the bbox as per camera's properties
  bbs = np.array([bb['corners'] for bb in bboxes]) - shift
  bbs = bbs @ rotation
  pbb = bbs @ K.T

  # remove z axis if any
  valid = np.any(pbb[...,-1] > 0, axis=-1)
  pbb /= pbb[...,-1:] + 1e-5
  uls = pbb.min(axis=1)
  lrs = pbb.max(axis=1)

  vboxes = []
  for v, ul, lr, bb in zip(valid, uls, lrs, bboxes):
      if v:
          x0, y0 = np.round(ul).astype(int)[:2]
          x1, y1 = np.round(lr).astype(int)[:2]
          x0 = np.clip(x0, a_min=0, a_max=1920)
          x1 = np.clip(x1, a_min=0, a_max=1920)
          y0 = np.clip(y0, a_min=0, a_max=1080)
          y1 = np.clip(y1, a_min=0, a_max=1080)

          # print(f"{idx}",x0, x1, y0, y1)

          if x1 > x0 and y1 > y0 and (x1-x0)*(y1-y0) < 1920*1080/2:
              roi = sem[y0:y1, x0:x1]
              flag = False
              if bb['class'] == [100]:
                  vboxes.append(([x0, y0, x1, y1],idmap[tuple(bb['class'])]))

  return to_pytorch(rgb_image, vboxes)

def to_pytorch(rgb, boxes):
    rgb = torch.from_numpy(rgb).permute(2,0,1).to(dtype=torch.float32)
    bbs = torch.tensor([bb[0] for bb in boxes], dtype=torch.float32)
    cls = torch.tensor([bb[1] for bb in boxes], dtype=torch.long)
    return rgb, bbs, cls

#visualise 
def get_figure(rgb, bboxes, cls):
  with torch.no_grad():
      fig, ax = plt.subplots()
      ax.imshow(rgb.permute(1,2,0).cpu().numpy())
      for bbox, cl in zip(bboxes, cls):
          x0, y0, x1, y1 = bbox.cpu().numpy()
          r = Rectangle([x0, y0], x1-x0, y1-y0, alpha=.5, color= np.array([  0,  0,142])/255.)
          ax.add_patch(r)
      plt.gca().set_axis_off()
      plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                  hspace = 0, wspace = 0)
      plt.margins(0,0)
      plt.gca().xaxis.set_major_locator(plt.NullLocator())
      plt.gca().yaxis.set_major_locator(plt.NullLocator())
  return fig

# Test bounding_boxes function
idx = 10
rgb, bbs, cls = bounding_boxes(idx,output_dir80m)

print(f"Image shape: {rgb.shape}")
print(f"Number of bounding boxes: {len(bbs)}")
print(f"Number of classes: {len(cls)}")
# fig = get_figure(rgb, bbs, cls)


#data transforms 
data_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomAutocontrast(0.5),
    transforms.Resize((192,640)), #this is the size monodepth2 uses
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


#creating the dataloader 
class CardetectDataset(Dataset):
  def __init__(self, output_dir, transform = None):
    self.output_dir = output_dir
    self.transform = transform

    self.rgb_dir = f'{output_dir}/rgb'


    # number of images
    img_file = sorted(os.listdir(self.rgb_dir))
    # print(img_file)
    idx_count = len(img_file)


    # self.rgb , self.bbs, self.cls = bounding_boxes(idx,output_dir)
    self.data = []
    for single_img in img_file:
      idx = int(single_img[:5])

      rgb , bbs, cls = bounding_boxes(idx,output_dir)
      # self.data.append((rgb, bbs, cls))
      if idx in [1000, 1500, 2000, 2999]:
        print(f"{idx} done")
            
      yolo_bbs = []
      for bb in bbs:
        x_center = (bb[0] + bb[2]) / 2 / rgb.shape[2]
        y_center = (bb[1] + bb[3]) / 2 / rgb.shape[1]
        width = (bb[2] - bb[0]) / rgb.shape[2]
        height = (bb[3] - bb[1]) / rgb.shape[1]
        yolo_bbs.append([x_center, y_center, width, height])

      self.data.append([rgb,yolo_bbs,cls])


  def __len__(self):
    return len(self.data)



  def __getitem__(self, idx):
    rgb, bbs, cls = self.data[idx]
    if self.transform:
      rgb = self.transform(rgb)
    return rgb, bbs, cls

    # print(rgbdir[0])

    # image_idx = int(rgbdir[indx][:5])
    # print(image_idx)



#comment data50 and data80 while running data20 to avoid high memory usage
data20 = CardetectDataset(output_dir20m)
data50 = CardetectDataset(output_dir50m)
data80 = CardetectDataset(output_dir80m)




#creating a yaml dataset 
def save_bounding_boxes_to_text(dataset, file_dir):
    #Looping throgh each image
    for idx, (_, bbs, cls) in enumerate(dataset.data):
      filename = f"{idx:05d}.txt"
      filepath = os.path.join(file_dir, filename)
      txt_data = []
      #Looping through each detection in that image
      for bb, cl in zip(bbs, cls):
        
          box = {
              # "bbox": bb.tolist() if isinstance(bb, torch.Tensor) else bb,
              "bbox" : [float(i) for i in bb],
              # "class": int(cl) if isinstance(cl, torch.Tensor) else cl,
              'class': 0,
              "id": len(txt_data)
          }
          # bbox_str = " ".join(f"{coord:.6f}" for coord in box["bbox"])  # Join bbox values as strings
          
          bbox_str = ([float(coord) for coord in bb])
          bbox_str.insert(0,int(box["class"]))

          # bbox_str = " ".join(f"float(coord):.6f}" for coord in bb) + f" {cl}\n"
          # box_str = f'{ {bbox_str}} {box["class"]} '
          
          txt_data.append(bbox_str)
          # filename = f"{idx:05d}.txt"
          # filepath = os.path.join(file_dir, filename)

      # Save the list of lists into a text file
      with open(filepath, "w") as file:
          for row in txt_data:
              
              # Join the float values as strings with a space separator and write to file
              file.write(" ".join(map(str, row)) + "\n")


#comment data50 and data80 while running data20 to avoid high memory usage


text_output_dir20 = '/home/rbhalekar/final_dataset/labels20'
text_output_dir50 = '/home/rbhalekar/final_dataset/labels50'
text_output_dir80 = '/home/rbhalekar/final_dataset/labels80'

os.makedirs(text_output_dir20, exist_ok=True)
save_bounding_boxes_to_text(data20, text_output_dir20)

os.makedirs(text_output_dir50, exist_ok=True)
save_bounding_boxes_to_text(data50, text_output_dir50)

os.makedirs(text_output_dir80, exist_ok=True)
save_bounding_boxes_to_text(data80, text_output_dir80)
 
   
