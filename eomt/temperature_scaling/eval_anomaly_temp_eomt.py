import os
import sys
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from argparse import ArgumentParser

#Check path correctness
sys.path.append("/content/drive/MyDrive/Colab Notebooks/AML_Project/MaskArchitectureAnomaly_CourseProject/eomt")

# Import your specific class (ensure it is the correct file)
# If the class is defined in the same training script, you might need to copy it here
# or import it from the .py file where you saved it.
from training.lightning_module import AnomalyDetectionModuleStreamingMetrics 

# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
class BinarySegmentationMetrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = self.fp = self.fn = self.tn = 0

    def update(self, pred_mask, gt_mask):
        valid = gt_mask != 255
        pred = pred_mask[valid].astype(bool)
        gt = gt_mask[valid].astype(bool)

        self.tp += np.logical_and(pred, gt).sum()
        self.fp += np.logical_and(pred, ~gt).sum()
        self.fn += np.logical_and(~pred, gt).sum()
        self.tn += np.logical_and(~pred, ~gt).sum()

    def get_metrics(self):
        eps = 1e-7
        iou = self.tp / (self.tp + self.fp + self.fn + eps)
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        return {"IoU": iou, "F1": f1, "Precision": precision, "Recall": recall}

# -----------------------------------------------------------------------------
# LOADING GROUND TRUTH
# -----------------------------------------------------------------------------
def load_ground_truth(pathGT, dataset_name):
    mask = Image.open(pathGT)
    ood_gts = np.array(mask).astype(np.int32)

    if "RoadAnomaly" in dataset_name:
        # In RoadAnomaly21: 0=background, 1=anomaly, 255=ignore
        # If your dataset uses 2 for anomaly, convert:
        ood_gts = np.where(ood_gts == 2, 1, ood_gts)
    elif "LostAndFound" in dataset_name:
        # 0=ignore/void, 1=road, >1=anomaly
        original = ood_gts.copy()
        ood_gts = np.zeros_like(original)
        ood_gts[original == 0] = 255   # Void
        ood_gts[original == 1] = 0     # Road
        ood_gts[original > 1] = 1      # Anomaly
    
    return ood_gts.astype(np.uint8)

# -----------------------------------------------------------------------------
# OPTIMIZED INFERENCE
# -----------------------------------------------------------------------------
def get_anomaly_score(model, img_path, device):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size

    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((640, 640)), # Consistent with AnomalyConfig
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        # We use DIRECTLY the function we wrote in the model
        # anomaly_score_640 has shape [640, 640] and is in range [0, 1]
        anomaly_score_640 = model._predict(img_tensor)
        
        # We resize back to the original resolution to calculate metrics on real pixels
        anomaly_score_full = F.interpolate(
            anomaly_score_640.unsqueeze(0).unsqueeze(0),
            size=(h, w),
            mode="bilinear",
            align_corners=False
        ).squeeze()

    return anomaly_score_full.cpu().numpy()