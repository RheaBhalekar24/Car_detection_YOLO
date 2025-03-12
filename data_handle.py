import os
import shutil
import random

# Step 1: Rename images in rgb and label folders
def rename_images(rgb_folder, label_folder, suffix):
    for folder in [rgb_folder, label_folder]:
        for idx, filename in enumerate(os.listdir(folder)):
            if folder == rgb_folder and filename.endswith('.jpg'):
                new_name = f"{str(idx).zfill(5)}_{suffix}.jpg"
            elif folder == label_folder and filename.endswith('.txt'):
                new_name = f"{str(idx).zfill(5)}_{suffix}.txt"
            else:
                continue

            src_path = os.path.join(folder, filename)
            dst_path = os.path.join(folder, new_name)

            os.rename(src_path, dst_path)

# Step 2: Merge rgb and label folders into a single `images` and `labels` folder
def merge_data(rgb_folders, label_folders, target_rgb_dir, target_label_dir):
    os.makedirs(target_rgb_dir, exist_ok=True)
    os.makedirs(target_label_dir, exist_ok=True)

    for rgb_folder in rgb_folders:
        for filename in os.listdir(rgb_folder):
            shutil.copy(os.path.join(rgb_folder, filename), os.path.join(target_rgb_dir, filename))

    for label_folder in label_folders:
        for filename in os.listdir(label_folder):
            shutil.copy(os.path.join(label_folder, filename), os.path.join(target_label_dir, filename))


def organize_dataset(image_dir, label_dir, output_dir):
    """
    Organizes the dataset into train and validation folders 
    
    Parameters:
    - image_dir: Path to the directory containing RGB images.
    - label_dir: Path to the directory containing label files.
    - output_dir: Path where the organized dataset will be saved.
    """
    # Create output directories
    train_images_dir = os.path.join(output_dir, 'train/images')
    train_labels_dir = os.path.join(output_dir, 'train/labels')
    val_images_dir = os.path.join(output_dir, 'val/images')
    val_labels_dir = os.path.join(output_dir, 'val/labels')

    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)

    # Split data into training and validation sets (80% train, 20% val)
    all_files = sorted(os.listdir(image_dir))
    num_files = len(all_files)
    
    # Calculate split index
    split_index = int(num_files * 0.8)

    # Copy files to train and val directories
    for i, filename in enumerate(all_files):
        image_path = os.path.join(image_dir, filename)
        label_path = os.path.join(label_dir, filename.replace('.jpg', '.txt'))

        if i < split_index:
            # Copy to training set
            shutil.copy(image_path, train_images_dir)
            shutil.copy(label_path, train_labels_dir)
        else:
            # Copy to validation set
            shutil.copy(image_path, val_images_dir)
            shutil.copy(label_path, val_labels_dir)


# Define your paths where you want the images to be stored 
rgb_folders = [
    '/home/rbhalekar/final_dataset/rgb20',
    '/home/rbhalekar/final_dataset/rgb50',
    '/home/rbhalekar/final_dataset/rgb80' 
]
label_folders = [
    '/home/rbhalekar/final_dataset/labels20' ,
    '/home/rbhalekar/final_dataset/labels50' ,
    '/home/rbhalekar/final_dataset/labels80' 
]

# Define your paths where you want the images to be stored 

target_rgb_dir = "/home/rbhalekar/merged_final_dataset/images"
target_label_dir = "/home/rbhalekar/merged_final_dataset/labels"

train_rgb_dir = "/home/rbhalekar/finally_merged_dataset/train/images"
train_label_dir = "/home/rbhalekar/finally_merged_dataset/train/labels"

val_rgb_dir = "/home/rbhalekar/finally_merged_dataset/val/images"
val_label_dir = "/home/rbhalekar/finally_merged_dataset/val/labels"

# Step 1: Rename

rename_images(rgb_folders[0],label_folders[0],20)
rename_images(rgb_folders[1],label_folders[1],50)
rename_images(rgb_folders[2],label_folders[2],80)

# # Merge data
merge_data(rgb_folders, label_folders, target_rgb_dir, target_label_dir)


# Example usage
image_directory = '/home/rbhalekar/merged_final_dataset/images'  
label_directory = '/home/rbhalekar/merged_final_dataset/labels'   
output_directory = '/home/rbhalekar/finally_merged_dataset1'  

organize_dataset(image_directory, label_directory, output_directory)
