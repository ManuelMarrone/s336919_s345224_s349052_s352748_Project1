import os
import yaml
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import numpy as np
from pycocotools.coco import COCO

def load_config(config_file):
    """
    Load configuration from YAML file.
    """
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def extract(coco_json_path, coco_img_dir, save_dir, categories=None, min_size=480):
    """
    Extract objects from COCO dataset, excluding Cityscapes classes.
    
    """
    # Initialize COCO API for instance annotations
    coco=COCO(coco_json_path)

    # Classes to exclude - these are Cityscapes classes also available in COCO
    exclude_classes = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'traffic light', 'stop sign']

    # Load category IDs based on requested categories
    if categories:
        # Filter requested categories, excluding Cityscapes classes
        valid_categories = [cat for cat in categories if cat not in exclude_classes]
        if not valid_categories:
            print(f"[ERROR] Tutte le categorie richieste sono escluse: {categories}")
            return
        
        categories_ids = coco.getCatIds(catNms=valid_categories)
    else:
        # Use all categories except excluded ones
        all_cats = coco.loadCats(coco.getCatIds())
        categories_ids = [cat['id'] for cat in all_cats if cat['name'] not in exclude_classes]
        print(f"Using all categories except: {exclude_classes}")

    # Get annotations that satisfy the given filtered categories
    ann_ids = coco.getAnnIds(catIds=categories_ids, iscrowd=False)
    annotations = coco.loadAnns(ann_ids)
    
    obj_id = 0
    for ann in annotations:
        
        #Bounding box formato: [x_min, y_min, width, height]
        x, y, w, h = map(int, ann['bbox'])
        
        # Skip objects smaller than minimum size threshold
        if h < min_size and w < min_size:
            continue
        
        # Load corresponding image
        image_info = coco.loadImgs(ann['image_id'])[0]
        imgae_path = Path(coco_img_dir) / image_info['file_name']
        if not imgae_path.exists():
            print(f"[WARN] Immagine non trovata: {imgae_path}")
            continue

        img = np.array(Image.open(imgae_path).convert("RGB"))

        # Generate object mask from annotation
        mask = coco.annToMask(ann)  # Binary mask: 0 or 1
        if mask.sum() == 0:
            continue

        # Extract object using bounding box
        obj_img = img[y:y+h, x:x+w]
        obj_mask = mask[y:y+h, x:x+w]
        
        # Skip if mask is empty after extracting
        if obj_mask.sum()==0:
            continue

        # Increment object counter
        obj_id += 1

        # Convert mask to grayscale image
        # Multiply by 255 to make visible: 0→black (background), 1→white (object)
        mask_image = Image.fromarray(obj_mask * 255, mode="L")
        
        # Convert object crop to PIL Image (already in RGB 0-255 range)
        obj_image = Image.fromarray(obj_img)
        
        # Save object and mask
        save_dir=Path(save_dir)
        save(obj_image, mask_image, save_dir, obj_id)

    print(f"[DONE] Extracted {obj_id} COCO objects saved in {save_dir}")


def save(obj_image, mask_image, save_dir,obj_id):
    """
    Save cropped object and its corresponding mask.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save object image
    file_path_obj = save_dir/ f"{obj_id}.png"  
    obj_image.save(file_path_obj)
    
    # Save mask image with "_mask" suffix
    file_path_mask = save_dir/ f"{obj_id}_mask.png"
    mask_image.save(file_path_mask)
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract COCO objects")
    parser.add_argument("--config", type=str, required=True, help="Path config YAML")
    args = parser.parse_args()

    config = load_config(args.config)   

    # Converti categories da stringa a lista
    categories = config.get('categories', None)
    if categories:
        if isinstance(categories, str):
            # Se è una stringa separata da virgole
            categories = [cat.strip() for cat in categories.split(',')]
        print(f"Requested categories: {categories}") 

    extract(
        coco_json_path=config['coco_json'],
        coco_img_dir=config['coco_img_dir'],
        save_dir=config['save_dir'],
        categories=categories,
        min_size=config.get('min_size', 480)
    )