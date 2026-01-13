import numpy as np
#import torch
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import os
import rasterio
import dataprocessing.rcsHandlingFunctions as rcs
import yaml
import os
from pathlib import Path

###########################################################################################################
###############This script creates a dataframe with image paths, dem paths, mask paths, images, dems, masks, etc.
###############The folders should be organized as follows:
###############root_dir/
###############    nested_folder_1/
###############        image_rgb.tif
###############        dem.tif
###############        mask.png
###############        rcs.tif
###############    nested_folder_2/
###########################################################################################################
###############FUNCTIONS###################################################################################
###########################################################################################################
def read_metadata(dem_file, print_info=True):
    with rasterio.open(dem_file) as src:
        # Access metadata
        metadata = src.meta
        width = src.width
        height = src.height
        crs = src.crs
        bounds = src.bounds
        transform = src.transform

    # Print the information
    print("Metadata:", metadata)
    print("Width:", width)
    print("Height:", height)
    print("CRS:", crs)
    print("Bounds:", bounds)
    print("Transform:", transform)


def make_dem_rgb(dem_data):
    if len(dem_data.shape) == 2:
        dem_data = np.expand_dims(dem_data, axis=-1)
    else:
        raise ValueError("DEM data should have 2 dimensions")
        return None 
    if dem_data.shape[-1] == 1:
        dem_data_rgb = np.repeat(dem_data, 3, axis=-1)
    return dem_data_rgb


#'.jpg', '.jpeg', '.png', '.tif'
def get_all_image_paths(root_dir, extensions=['rgb.tif']):
    image_paths = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                image_paths.append(os.path.join(subdir, file))
    return image_paths


def read_images(image_paths):
    images = []
    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            images.append((path, img))
    return images

def is_all_black(image):
    return np.all(image == 0)

def find_elements_with_substring(strings, substring):
    return [s for s in strings if substring in s]

# Function to find bounding boxes for each group of disconnected white pixels
def find_bounding_boxes(mask):
    # Ensure the mask is an 8-bit image.
    if mask.dtype != "uint8":
        # If mask values are in range 0-1, scale them by 255
        if mask.max() <= 1:
            mask_uint8 = (mask * 255).astype('uint8')
        else:
            mask_uint8 = mask.astype('uint8')
    else:
        mask_uint8 = mask
    
    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    H, W = mask.shape
    
    # Compute bounding boxes for each contour
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]

    return bounding_boxes

# Function to check if two bounding boxes intersect
def do_boxes_intersect(box1, box2):
    x1_min, y1_min, w1, h1 = box1
    x1_max, y1_max = x1_min + w1, y1_min + h1
    x2_min, y2_min, w2, h2 = box2
    x2_max, y2_max = x2_min + w2, y2_min + h2
    
    # Check if there is an overlap
    if x1_min < x2_max and x1_max > x2_min and y1_min < y2_max and y1_max > y2_min:
        return True
    return False

# Function to merge two bounding boxes
def merge_boxes(box1, box2):
    x1_min, y1_min, w1, h1 = box1
    x1_max, y1_max = x1_min + w1, y1_min + h1
    x2_min, y2_min, w2, h2 = box2
    x2_max, y2_max = x2_min + w2, y2_min + h2
    
    # Find the coordinates of the merged bounding box
    x_min = min(x1_min, x2_min)
    y_min = min(y1_min, y2_min)
    x_max = max(x1_max, x2_max)
    y_max = max(y1_max, y2_max)
    
    # Compute the width and height of the merged bounding box
    w = x_max - x_min
    h = y_max - y_min
    
    return (x_min, y_min, w, h)

# Check for overlapping bounding boxes and merge them
def merge_overlapping_boxes(bounding_boxes):
    merged_boxes = []
    while bounding_boxes:
        box = bounding_boxes.pop(0)
        i = 0
        while i < len(bounding_boxes):
            if do_boxes_intersect(box, bounding_boxes[i]):
                box = merge_boxes(box, bounding_boxes.pop(i))
            else:
                i += 1
        merged_boxes.append(box)
    return merged_boxes

def increase_bounds(bounding_boxes, H, W, increase_by=20, augment=True):
    # Increase the bounding box size by 20 pixels in all directions

    expanded_bounding_boxes = []
    for (x, y, w, h) in bounding_boxes:
        if augment:
            # Randomly increase by up to 'increase_by' pixels
            x_min = max(0, x - np.random.randint(0, increase_by))
            y_min = max(0, y - np.random.randint(0, increase_by))
            x_max = min(W, x + w + np.random.randint(0, increase_by))
            y_max = min(H, y + h + np.random.randint(0, increase_by))
        else:
            # Increase by a fixed 'increase_by' pixels
            x_min = max(0, x - increase_by/2)
            y_min = max(0, y - increase_by/2)
            x_max = min(W, x + w + increase_by/2)
            y_max = min(H, y + h + increase_by/2)
        expanded_bounding_boxes.append((x_min, y_min, x_max - x_min, y_max - y_min))
    return expanded_bounding_boxes

def create_bounding_boxes(mask, increase_by=20, augment=True):
    H, W = mask.shape
    bounding_boxes = find_bounding_boxes(mask)
    expanded_bounding_boxes = increase_bounds(bounding_boxes, H, W, increase_by=increase_by, augment=augment)
    num_of_boxes = len(expanded_bounding_boxes)
    while(True):
        expanded_bounding_boxes = merge_overlapping_boxes(expanded_bounding_boxes)
        if (num_of_boxes == len(expanded_bounding_boxes)):
            break
        else:
            num_of_boxes = len(expanded_bounding_boxes)
    return expanded_bounding_boxes


def read_images_dem_masks_resize(img_dem_masks_paths):
    data = []
    for img_path, dem_path, mask_path in img_dem_masks_paths:
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        # Resize the image
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
        with rasterio.open(dem_path) as src:
            dem_data = src.read(1)
            dem_data = cv2.resize(dem_data, (512, 512), interpolation=cv2.INTER_LINEAR)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        rcs_path = img_path.replace('rgb.tif', 'rcs.tif')
        rcs_data = rcs.read_rcs_image(rcs_path)
        if mask is not None:
            # Resize the mask
            mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
            data.append({'image_path': img_path,
                         'image': img, 
                         'mask_path': mask_path, 
                         'mask': mask, 
                         'dem_path': dem_path, 
                         'dem': dem_data,
                         'empty_mask': is_all_black(mask), 
                         'no_mask': False,
                         'boxes': create_bounding_boxes(mask),
                         'rcs': rcs_data})
        else:
            data.append({'image_path': img_path,
                         'image': img, 
                         'mask_path': mask_path, 
                         'mask': mask, 
                         'dem_path': dem_path, 
                         'dem': dem_data,
                         'empty_mask': True, 
                         'no_mask': True,
                         'boxes': None,
                         'rcs': rcs_data})
    return data

###########################################################################################################
###############AVALANCHE SEGMENTATION DATAFRAME CREATION###################################################
###########################################################################################################

# Main block to prevent code from running on import
if __name__ == "__main__":
    

    # 1. Get the path of the script
    current_file = Path(__file__).resolve() # src/training/your_script.py

    # 2. Go up one level to 'src', then into 'config'
    config_path = current_file.parent.parent / "config" / "config_general.yaml"

    # 3. Load the YAML
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 4. Resolve the root of the project (one level above 'src')
    # This ensures that "./data" in the YAML is interpreted relative to the Project_Root
    PROJECT_ROOT = current_file.parent.parent.parent
    os.chdir(PROJECT_ROOT) 

    # Extract paths from YAML
    DATA_DIR = config['paths']['data']
    CHECKPOINT_DIR = config['paths']['checkpoints']
    SAM_CHECKPOINT = config['paths']['sam_checkpoint']
    
    # Path to the root directory containing nested folders with masks
    mask_root_dir = os.path.join(DATA_DIR, 'masks')

    # Get all masks paths 
    mask_paths = get_all_image_paths(mask_root_dir, ['mask.png'])


    # Path to the root directory containing nested folders with dem_images and images
    root_dir = os.path.join(DATA_DIR, 'images')

    # Get all dem paths
    dem_paths = get_all_image_paths(root_dir, ['dem.tif'])

    # Get all image paths
    image_paths = get_all_image_paths(root_dir)

    img_dem_paths = zip(image_paths, dem_paths)
    img_dem_masks_paths= []
    for img_path, dem_path in img_dem_paths:
        basename = os.path.basename(img_path)
        basename = basename.replace('rgb.tif','')
        mask_path = find_elements_with_substring(mask_paths, basename)
        if(len(mask_path) == 1):
            img_dem_masks_paths.append((img_path, dem_path, mask_path[0]))
        else:
            img_dem_masks_paths.append((img_path, dem_path, None))
            #print(f"Mask not found for image {basename}")

    print(f"Total images with dem and masks: {len(img_dem_masks_paths)}")
    print(img_dem_masks_paths[0])

    data = read_images_dem_masks_resize(img_dem_masks_paths)

    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(data)

    # Save the DataFrame to a file (optional)
    df.to_pickle('dataframe_avalanches_resize.pkl')
