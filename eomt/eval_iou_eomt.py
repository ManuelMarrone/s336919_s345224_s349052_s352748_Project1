import os
import sys
import torch
import torch.nn.functional as F
import importlib
import time
import yaml
from PIL import Image
from argparse import ArgumentParser
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize
from torchvision.transforms import ToTensor, ToPILImage

from dataset import cityscapes
from transform import Relabel, ToLabel, Colorize
from iouEval import iouEval, getColorEntry

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EVAL_ROOT = os.path.abspath(os.path.dirname(__file__))
EOMT_ROOT = os.path.join(PROJECT_ROOT, "eomt")

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, EVAL_ROOT)  
sys.path.insert(0, EOMT_ROOT)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open("eomt/configs/dinov2/cityscapes/semantic/eomt_base_640.yaml") as f:
    config = yaml.safe_load(f)

NUM_CHANNELS = 3
NUM_CLASSES = 19 # Cityscapes semantic classes
IMG_SIZE = (512, 1024)   # Cityscapes standard resolution


# Image preprocessing transforms
image_transform = ToPILImage()
input_transform_cityscapes = Compose([
    Resize(512, Image.BILINEAR),
    ToTensor(),
])
target_transform_cityscapes = Compose([
    Resize((512, 1024), Image.NEAREST),
    ToLabel(), # Convert to label format
])

def main(args):

    # Load encoder
    encoder_cfg = config["model"]["init_args"]["network"]["init_args"]["encoder"]
    encoder_module_name, encoder_class_name = encoder_cfg["class_path"].rsplit(".", 1)
    encoder_cls = getattr(importlib.import_module(encoder_module_name), encoder_class_name)
    encoder = encoder_cls(img_size=IMG_SIZE, **encoder_cfg.get("init_args", {}))

    # Load network
    network_cfg = config["model"]["init_args"]["network"]
    network_module_name, network_class_name = network_cfg["class_path"].rsplit(".", 1)
    network_cls = getattr(importlib.import_module(network_module_name), network_class_name)
    network_kwargs = {k: v for k, v in network_cfg["init_args"].items() if k != "encoder"}
    network = network_cls(
        masked_attn_enabled=False,
        num_classes=NUM_CLASSES,
        encoder=encoder,
        **network_kwargs,
    ).to(device).eval()

    
    # Load pre-trained checkpoint
    ckpt_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "CourseProjectAnomaly",
        "eomt_cityscapes.bin"
    )

    # Clean up state dict: remove "network." prefix and pos_embed keys
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)

    state_dict = {
        k.replace("network.", ""): v
        for k, v in state_dict.items()
    }

    state_dict = {
        k: v for k, v in state_dict.items()
        if "pos_embed" not in k
    }

    # Load weights into model
    network.load_state_dict(state_dict, strict=False)

    
    # Setup Dataset and Loader
    # Create Cityscapes dataset:
    # - Loads RGB images from args.datadir/leftImg8bit/{subset}/
    # - Loads ground truth semantic labels from args.datadir/gtFine/{subset}/
    # - input_transform: preprocesses images (resize to 512x1024 → convert to tensor)
    # - target_transform: preprocesses labels (resize to 512x1024 → convert to class indices)
    # DataLoader creates batches and loads data in parallel with num_workers processes
    loader = DataLoader(
        cityscapes(args.datadir, input_transform_cityscapes, target_transform_cityscapes, subset=args.subset), 
        num_workers=args.num_workers, 
        batch_size=args.batch_size,
        shuffle=False)
    print("Dataset size:", len(loader.dataset))
    print("Images:", len(loader.dataset.filenames))
    print("GT:", len(loader.dataset.filenamesGT))

    # 20 classes: 19 semantic + 1 ignore class (index 19)
    iouEvalVal = iouEval(20, ignoreIndex=19)

    start = time.time()

    for step, (images, labels, filename, filenameGt) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            mask_logits_per_layer, class_logits_per_layer = network(images)

        mask_logits  = mask_logits_per_layer[-1]   # [B, Q, H, W]
        class_logits = class_logits_per_layer[-1]  # [B, Q, C+1]

        class_probs = F.softmax(class_logits, dim=-1)
        class_probs = class_probs[..., :-1]  # togli no-object
        mask_probs  = torch.sigmoid(mask_logits)

        # For pixel (h,w) and class c: sum_q [class_probs[q,c] * mask_probs[q,h,w]]
        pixel_class_probs = torch.einsum(
            "bqc,bqhw->bchw", # (batch, queries, classes) × (batch, queries, H, W) → (batch, classes, H, W)
            class_probs,
            mask_probs
        ) # Output: [B, C, H, W]

        # Resize spatial dimensions (H, W) to match label resolution
        pixel_class_probs = F.interpolate(
            pixel_class_probs,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        if labels.dim() == 3:
            # Shape: [B, H, W] → [B, 1, H, W]
            # Add channel dimension (1) to match prediction format
            labels = labels.unsqueeze(1)

        # Map void class (255) to ignore index (19) for IoU calculation
        # Cityscapes uses 255 for unlabeled pixels (image borders, unclear regions)
        # IoU evaluator expects ignore class to be index 19
        labels_for_eval = labels.clone()
        labels_for_eval[labels_for_eval == 255] = 19

        pred = pixel_class_probs.argmax(dim=1, keepdim=True)

        # Add the current batch predictions and ground truth labels to the IoU evaluator
        # This accumulates results across all batches for final metric calculation
        iouEvalVal.addBatch(pred, labels_for_eval)

        filenameSave = filename[0].replace("\\","/")

        print (step, filenameSave)

    # After processing all batches, compute the final IoU metrics
    # iouVal: mean IoU across all classes
    # iou_classes: per-class IoU values as a tensor of shape [NUM_CLASSES]
    iouVal, iou_classes = iouEvalVal.getIoU()

    iou_classes_str = []
    for i in range(iou_classes.size(0)):
        iouStr = getColorEntry(iou_classes[i])+'{:0.2f}'.format(iou_classes[i]*100) + '\033[0m'
        iou_classes_str.append(iouStr)

    print("---------------------------------------")
    print("Took ", time.time()-start, "seconds")
    print("=======================================")
    
    # Print IoU for each of the 19 Cityscapes classes with their names
    print("Per-Class IoU:")
    print(iou_classes_str[0], "Road")
    print(iou_classes_str[1], "sidewalk")
    print(iou_classes_str[2], "building")
    print(iou_classes_str[3], "wall")
    print(iou_classes_str[4], "fence")
    print(iou_classes_str[5], "pole")
    print(iou_classes_str[6], "traffic light")
    print(iou_classes_str[7], "traffic sign")
    print(iou_classes_str[8], "vegetation")
    print(iou_classes_str[9], "terrain")
    print(iou_classes_str[10], "sky")
    print(iou_classes_str[11], "person")
    print(iou_classes_str[12], "rider")
    print(iou_classes_str[13], "car")
    print(iou_classes_str[14], "truck")
    print(iou_classes_str[15], "bus")
    print(iou_classes_str[16], "train")
    print(iou_classes_str[17], "motorcycle")
    print(iou_classes_str[18], "bicycle")
    print("=======================================")
    iouStr = getColorEntry(iouVal)+'{:0.2f}'.format(iouVal*100) + '\033[0m'
    print ("MEAN IoU: ", iouStr, "%")

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--subset', default="val")  
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)

    main(parser.parse_args())