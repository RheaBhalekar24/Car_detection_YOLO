# Car_detection_YOLO
Car detection using YOLO and trained on aerial data 
This is the readme file for the part1 of object detection 

Requirements :
Python 3.8 and above 
latest versions of torch , numpy 
open cv headless 

Files:

1) ml_project_part1.py = creates the dataloader and converts 3D bounding boxes into 2D
2) data_handle.py = combines 20m , 50m and 80m data into one folder , annotates the height to file name and performs train-val split 

3) visualise.py =  visualise any single image from the dataset

4) part1_train.py = train  the model

5) part1_eval.py = evaluate the model

6) images10_predict.py = predict 10 images randomly selected 

7) output_images = stores predicted images 

8) yolo11l.pt = yolo model used during training 
9) trained_yolo11l.pt = yolo model used for prediction 
10) yaml file : insert yaml file into your dataset once created 


Process:

provide path links to the main directory of the dataset and the directories to individual rgb images of varying heights i.e 20m,50m and 80m
 in the ml_project_part1.py and for  data_handle.py the directory where you want the dataset to be stored  
and the directories to individual rgb images of varying heights i.e 20m,50m and 80m

create directory for your dataset 
store yaml file in your dataaset 

add paths to the yolo model as per your directory 


Run: 
Run in the following order of files mentioned above 




