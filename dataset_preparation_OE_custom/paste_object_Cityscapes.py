import os
import yaml
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import random
import cv2

 
def load_config(config_file):
    """
    Load configuration from YAML file
    """
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def load_coco_objects(obj_dir):
    """
    Load COCO objects images with their corresponding masks
    """
    obj_dir = Path(obj_dir)
    
    # Get all images except mask files (exclude *_mask.png)
    images = sorted(obj_dir.glob("*[!_mask].png"))  

    objects=[]
    for img in images:
        # For each image load corresponding mask
        mask_path = img.parent / f"{img.stem}_mask.png"

        mask = np.array(Image.open(mask_path))
        mask = (mask > 0).astype(np.uint8)  # Convert from 0-255 to binary 0-1

        image = np.array(Image.open(img).convert("RGB"))

        objects.append((image, mask))
    
    # Return a list of objects, each object is (image,mask)
    return objects

def blend_object(city_roi, obj_img, obj_mask, mode='gaussian', sigma=5):
    """
    Blend object with background using different modes.
    
    """
    
    if mode == 'gaussian':
        # Convert binary mask to float, necessary for Gaussian blurred 
        mask_float = obj_mask.astype(np.float32)
        # Blur the mask: sigma controls blur strength: higher sigma = softer transition
        mask_blur = cv2.GaussianBlur(mask_float, (0,0), sigmaX=sigma, sigmaY=sigma) # Shape: (H, W) → (H, W, 1)
        mask_blur = mask_blur[..., None]  # Shape (H,W,1) for RGB broadcasting
        
        # Result: smooth transition from background to object
        city_roi[:] = (city_roi*(1-mask_blur) + obj_img*mask_blur).astype(np.uint8)
    
    else:
        raise ValueError(f"Unknown blend mode '{mode}.")
    
    return city_roi

def paste_object(city_img, city_mask, obj_img, obj_mask,scale_range, rotation_range, paste_prob,blend_mode='none', blend_sigma=5, max_tries=50):
    """
    Paste a COCO object on a Cityscapes image with random transformations.
    """
    
    # Skip pasting based on paste_prob value
    if random.random() > paste_prob:
        return city_img, city_mask  

    H, W = city_img.shape[:2]
    h, w = obj_img.shape[:2]

    # Random scaling of object size
    scale = random.uniform(*scale_range)
    new_h, new_w = int(h*scale), int(w*scale)
    obj_img = cv2.resize(obj_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR) ## Resize object image with linear interpolation for smooth, natural appearance
    obj_mask = cv2.resize(obj_mask.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST) ## Resize mask with nearest neighbor to preserve binary values (0/1 only, no intermediates)


    # Rotation with a random angle selected from a range
    angle = random.uniform(rotation_range[0], rotation_range[1])
    center = (new_w / 2, new_h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0) #scale=1 original dimension 
    
    # Calculate new dimensions to fit rotated object
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w_rot = int((new_h * sin) + (new_w * cos))
    new_h_rot = int((new_h * cos) + (new_w * sin))
    
    # Adjust rotation matrix for new center
    M[0, 2] += (new_w_rot / 2) - center[0]
    M[1, 2] += (new_h_rot / 2) - center[1]
    
    # Apply rotation
    obj_img = cv2.warpAffine(obj_img, M, (new_w_rot, new_h_rot), 
                              flags=cv2.INTER_LINEAR, 
                              borderMode=cv2.BORDER_CONSTANT, 
                              borderValue=(0, 0, 0)) # RGB, Zero (background)
    obj_mask = cv2.warpAffine(obj_mask, M, (new_w_rot, new_h_rot), 
                               flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=0) # Zero (background)

    # Check if object fits in image
    h, w = obj_img.shape[:2]
    x_max = W - w
    y_max = H - h
    if x_max <= 0 or y_max <= 0:
        # Object too large, skip it
        return city_img, city_mask  

    # Convert mask to boolean for indexing: 0→False(no object), any non-zero→True(object)
    roi_mask = obj_mask.astype(bool) 
    
    #Try to find a non-overlapping position
    for attempt in range(max_tries):
        x = random.randint(0, x_max)
        y = random.randint(0, y_max)
        
        # Check overlap with existing anomalies
        overlap = city_mask[y:y+h, x:x+w][roi_mask].sum() 
        if overlap == 0:
            # No overlap found, paste object
            roi = city_img[y:y+h, x:x+w]
            blended_roi = blend_object(roi, obj_img, obj_mask, 
                                                  mode=blend_mode, 
                                                  sigma=blend_sigma)

            # Update image with blended object
            city_img[y:y+h, x:x+w] = blended_roi
            
            # Set anomaly mask to 1 following the shape of the object
            city_mask[y:y+h, x:x+w][roi_mask] = 1

            return city_img, city_mask

    # Failed to find non-overlapping position after max_tries, return Cityscapes image without the object
    return city_img, city_mask


def oe_dataset_generate(config):
    """
    Generate Outlier Exposure dataset by pasting COCO objects on Cityscapes images.
    """
    # Load all images of Cityscapes
    city_img_dir = Path(config['city_img_dir'])
    city_images = sorted(city_img_dir.rglob("*.png"))

    # Save the path for the final images and mask
    save_img_dir = Path(config['save_img_dir'])
    save_mask_dir = Path(config['save_mask_dir'])
    
    save_img_dir.mkdir(parents=True, exist_ok=True)
    save_mask_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all COCO objects once
    coco_objects = load_coco_objects(config['coco_obj_dir'])


    for img in city_images:
        
        #For each image of Cityscapes
        city_img = np.array(Image.open(img).convert("RGB"))
        H, W = city_img.shape[:2]
        city_mask = np.zeros((H, W), dtype=np.uint8)  # 
        
        # Select only a random number of object 
        num_obj = random.randint(*config['objects_per_image'])
        selected_objects = random.sample(coco_objects, num_obj)

        for obj_img, obj_mask in selected_objects:
            #Paste each object and mask
            city_img, city_mask = paste_object(
                city_img, city_mask, obj_img, obj_mask,
                config['scale_range'], 
                config['rotation_range'],
                config['paste_prob'],
                config.get('blend_mode', 'gaussian'),
                config.get('blend_sigma', 5)
            )

        # Save processed image and mask
        Image.fromarray(city_img).save(save_img_dir / img.name)
        Image.fromarray(city_mask*255).save(save_mask_dir / img.name)
            
    print("[DONE] Dataset generated!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paste COCO objects in Cityscapes images")
    parser.add_argument("--config", type=str, required=True, help="Path config YAML")
    args = parser.parse_args()

    config = load_config(args.config)
    oe_dataset_generate(config)