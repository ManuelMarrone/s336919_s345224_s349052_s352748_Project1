import os
import sys
import glob
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from argparse import ArgumentParser
from torchvision.transforms.v2 import functional as F_v2

# -----------------------------------------------------------------------------
# SETUP
# -----------------------------------------------------------------------------

sys.path.append(
    "/content/drive/MyDrive/temperature/MaskArchitectureAnomaly_CourseProject-main/MaskArchitectureAnomaly_CourseProject-main"
)

from training.lightning_module import LightningModule

# -----------------------------------------------------------------------------
# METRICS
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
        
        return {
            "IoU": iou,
            "F1": f1,
            "Precision": precision,
            "Recall": recall,
            
        }

# -----------------------------------------------------------------------------
# GROUND TRUTH
# -----------------------------------------------------------------------------
def load_ground_truth(pathGT, dataset_name):
        mask = Image.open(pathGT)
        # Convert in array
        ood_gts = np.array(mask).astype(np.int32)
    
        # ---------------------------------------------------------
        # MANAGE DIFFERENT DATASET FORMATS
        # ---------------------------------------------------------
    
        if "RoadAnomaly" in dataset_name:
            # RoadAnomaly: usually 2 is anomaly
            ood_gts = np.where(ood_gts == 2, 1, ood_gts)
    
        elif "FS_LostFound_full" in dataset_name:
            
            if np.max(ood_gts) > 1 and np.max(ood_gts) != 255:
                 # Original LostAndFound format (unprocessed)
                 original = ood_gts.copy()
                 ood_gts = np.zeros_like(original)
                 ood_gts[original == 0] = 255   # Void
                 ood_gts[original == 1] = 0     # Road (Normal)
                 ood_gts[(original > 1) & (original < 201)] = 1 # Obstacle
            else:
            
                 pass
    
    
    
      
    
        return ood_gts.astype(np.uint8)
# -----------------------------------------------------------------------------
# INFERENCE
# -----------------------------------------------------------------------------

def predict(model, img_path ):
        device = next(model.parameters()).device  #for gpu 
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        
        
        
        img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float()  # [3, H, W]
        
        img = F_v2.resize(
                    img,
                    (640, 640),
                    interpolation=F_v2.InterpolationMode.BILINEAR
                )
                
        img = img.to(device) #for gpu              
        with torch.no_grad():       
            mask_logits_per_block, class_logits_per_block = model(img)
        mask_logits = mask_logits_per_block[-1]   # [B, Q, 160, 160]
        class_logits = class_logits_per_block[-1] # [B, Q, C]

        # 1. IMMEDIATE UPSAMPLING (On Logits)
        # Upsample logits to 640x640 BEFORE any other operation
        mask_logits_high = F.interpolate(
            mask_logits,
            size=(h,w),
            mode='bilinear',
            align_corners=False
        ) # Result: [B, Q, 640, 640]

        # 2. Calculate probabilities on the large masks
        mask_prob = torch.sigmoid(mask_logits_high)

        class_prob = torch.softmax(class_logits, dim=-1)
        max_knownprob = class_prob[..., 0]

        # 4. Final fusion at 640x640
        pixel_known_conf = torch.max(
            mask_prob * max_knownprob[:, :, None, None],
            dim=1
        )[0]

        anomaly_score =  1 - pixel_known_conf # Already at 640x640


        return anomaly_score.squeeze()

# -----------------------------------------------------------------------------
# EVALUATION
# -----------------------------------------------------------------------------
def evaluate(ckpt, input_dir, gt_dir, dataset_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LightningModule.load_from_checkpoint(ckpt, strict=False)
    model.to(device).eval()

    # FIX device per pixel_mean / pixel_std
    model.network.encoder.pixel_mean = model.network.encoder.pixel_mean.to(device)
    model.network.encoder.pixel_std  = model.network.encoder.pixel_std.to(device)
    
    metrics = BinarySegmentationMetrics()
    all_scores = []
    all_gts = []

    # -------------------------------------------------------------------------
    # MODIFY DEBUG PATH
    # -------------------------------------------------------------------------
    ckpt_filename = os.path.splitext(os.path.basename(ckpt))[0]
    base_checkpoints_path = "/content/drive/MyDrive/temperature/MaskArchitectureAnomaly_CourseProject-main/MaskArchitectureAnomaly_CourseProject-main/eomt/checkpoints"
    debug_dir = os.path.join(base_checkpoints_path, dataset_name, f"{ckpt_filename}_DEBUG_VIS")
    os.makedirs(debug_dir, exist_ok=True)
    
    print(f"Directory debug visualizations: {debug_dir}")

    img_paths = sorted(glob.glob(os.path.join(input_dir, "*.*")))

    # -------------------------------------------------------------------------
    # FIRST PASS: score collection
    # -------------------------------------------------------------------------

    for img_path in tqdm(img_paths, desc="Collecting scores"):
        name = os.path.basename(img_path)
        gt_path = os.path.join(gt_dir, os.path.splitext(name)[0] + ".png")
        if not os.path.exists(gt_path):
            continue

        gt_mask = load_ground_truth(gt_path, dataset_name)

        anomaly_score = predict(model, img_path).detach().cpu().numpy()

        valid = gt_mask != 255
        all_scores.append(anomaly_score[valid])
        all_gts.append(gt_mask[valid])

    all_scores = np.concatenate(all_scores)
    all_gts = np.concatenate(all_gts)

    
    # -------------------------------------------------------------------------
    # ROC + PR CURVE + THRESHOLD (MODIFIED)
    # -------------------------------------------------------------------------

    from sklearn.metrics import roc_curve, auc, precision_recall_curve

    # Calculate AUROC
    fpr, tpr, _ = roc_curve(all_gts, all_scores)
    auroc = auc(fpr, tpr)
    print(f"\nPixel-wise AUROC: {auroc:.4f}")

    # --- NEW: Calculate FPR95 ---
    # Find the first index where TPR (Recall) >= 0.95
    idx_tpr95 = np.where(tpr >= 0.95)[0]
    if len(idx_tpr95) > 0:
        fpr95 = fpr[idx_tpr95[0]]
    else:
        fpr95 = 1.0 # Edge case if 95% is never reached
    print(f"Pixel-wise FPR95: {fpr95:.4f}")
    # -----------------------------

    # Calculate Precision-Recall curves
    precision, recall, thresholds = precision_recall_curve(all_gts, all_scores)
    
    # --- NEW: Calculate AUPRC ---
    # AUPRC is the area under the Precision(y) - Recall(x) curve
    auprc = auc(recall, precision)
    print(f"Pixel-wise AUPRC: {auprc:.4f}")
    # -----------------------------

    f1_curve = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_curve)
    best_threshold = thresholds[best_idx]

    print(f"Best F1: {f1_curve[best_idx]:.4f} @ threshold={best_threshold:.4f}")

    # -------------------------------------------------------------------------
    # SECOND PASS: metrics + visual debug
    # -------------------------------------------------------------------------

    saved = 0
    for img_path in tqdm(img_paths, desc="Final evaluation"):
        name = os.path.basename(img_path)
        gt_path = os.path.join(gt_dir, os.path.splitext(name)[0] + ".png")
        if not os.path.exists(gt_path):
            continue

        gt_mask = load_ground_truth(gt_path, dataset_name)
        anomaly_score = predict(model, img_path)

        pred_mask_tensor = (anomaly_score > best_threshold).to(torch.uint8)
        pred_mask = pred_mask_tensor.cpu().numpy()
        
        metrics.update(pred_mask, gt_mask)

        # ---------------- DEBUG VISUAL ----------------
        if saved < 10:
            viz_pred = (pred_mask * 255).astype(np.uint8)
            viz_gt = (gt_mask * 255).astype(np.uint8)
            viz_gt[gt_mask == 255] = 128  
            
            img_orig = Image.open(img_path).resize(
                (pred_mask.shape[1], pred_mask.shape[0])
            )
            
            Image.fromarray(viz_pred).save(os.path.join(debug_dir, f"pred_{name}.png"))
            Image.fromarray(viz_gt).save(os.path.join(debug_dir, f"gt_{name}.png"))
            img_orig.save(os.path.join(debug_dir, f"orig_{name}.jpg"))
            
            if saved == 0:
                print(f"Debug images saved in: {debug_dir}")
            saved += 1

    # -------------------------------------------------------------------------
    # FINAL RESULTS
    # -------------------------------------------------------------------------

    results = metrics.get_metrics()
    print(f"\nFinal results for {dataset_name} and {ckpt}:")
    
    # Add AUROC, AUPRC, and FPR95 to the results dictionary
    results["AUROC"] = auroc
    results["AUPRC"] = auprc
    results["FPR95"] = fpr95  # <--- ADDED
    
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    # -------------------------------------------------------------------------
    # SALVATAGGIO SU FILE TXT (MODIFICATO)
    # -------------------------------------------------------------------------
    output_txt_dir = os.path.dirname(debug_dir)
    output_txt_path = os.path.join(output_txt_dir, f"results_{ckpt_filename}.txt")

    with open(output_txt_path, "w") as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Checkpoint Path: {ckpt}\n")
        f.write(f"Checkpoint Name: {ckpt_filename}\n")
        f.write("-" * 50 + "\n")
        f.write(f"Pixel-wise AUROC: {auroc:.4f}\n")
        f.write(f"Pixel-wise AUPRC: {auprc:.4f}\n")
        f.write(f"Pixel-wise FPR95: {fpr95:.4f}\n") 
        f.write(f"Best Threshold (from PR-Curve): {best_threshold:.4f}\n")
        f.write("-" * 50 + "\n")
        f.write("Final Metrics (Calculated at Best Threshold):\n")
        
        for k, v in results.items():
            if k not in ["AUROC", "AUPRC", "FPR95"]: 
                 f.write(f"{k}: {v:.4f}\n")
    
    print(f"Risultati salvati in: {output_txt_path}")

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--dataset_name", default="RoadAnomaly21")
    args = parser.parse_args()

    evaluate(args.ckpt, args.input_dir, args.gt_dir, args.dataset_name)