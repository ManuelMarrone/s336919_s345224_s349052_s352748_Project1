import os
import sys
import glob
import random
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from argparse import ArgumentParser
from torchvision.transforms import Compose, Resize, ToTensor
from sklearn.metrics import average_precision_score, roc_curve
from models.vit import ViT
from models.eomt import EoMT
from training.lightning_module import LightningModule

# -----------------------------------------------------------------------------
# SETUP PATH 
# -----------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(CURRENT_DIR, "..")
EOMT_ROOT = os.path.join(PROJECT_ROOT, "eomt")

if EOMT_ROOT not in sys.path:
    sys.path.insert(0, EOMT_ROOT)
# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

NUM_CLASSES = 19              
IMG_SIZE = (1024, 1024)       
NUM_QUERIES = 100
NUM_BLOCKS = 3
BACKBONE_NAME = "vit_base_patch14_reg4_dinov2"
SEED = 100

# Setup 
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True # Force CUDA to use deterministic algorithms for reproducibility
torch.backends.cudnn.benchmark = True # Optimize execution speed for fixed input sizes

# Input/Target Transformations
input_transform = Compose([
    Resize(IMG_SIZE, Image.BILINEAR), # Resize RGB image using bilinear interpolation
    ToTensor(),                       # Convert to PyTorch tensor (0-1 float range)
])

target_transform = Compose([
    Resize(IMG_SIZE, Image.NEAREST), # Resize mask using nearest neighbor to preserve integer class IDs
])

# -----------------------------------------------------------------------------
# UTILITY
# -----------------------------------------------------------------------------


def fpr_at_95_tpr(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute False Positive Rate (FPR) when the True Positive Rate (TPR) is at 95%.
    """
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    idxs = np.where(tpr >= 0.95)[0]
    if len(idxs) == 0:
        return 1.0 
    return float(fpr[idxs[0]])

def load_eomt(ckpt_path: str, device: torch.device) -> EoMT:
    # Initialize ViT backbone and EoMT network architecture
    print(f"--> Initializing Backbone: {BACKBONE_NAME}")
    encoder = ViT(img_size=IMG_SIZE, backbone_name=BACKBONE_NAME)

    print(f"--> Initializing EoMT Network (Classes: {NUM_CLASSES}, Queries: {NUM_QUERIES})")
    network = EoMT(
        encoder=encoder,
        num_classes=NUM_CLASSES,
        num_q=NUM_QUERIES,
        num_blocks=NUM_BLOCKS,
        masked_attn_enabled=True,
    )

    # Full initialization with all required parameters to ensure compatibility with LightningModule
    lm = LightningModule(
        network=network,
        img_size=IMG_SIZE,
        num_classes=NUM_CLASSES,
        attn_mask_annealing_enabled=False,  
        attn_mask_annealing_start_steps=0,     
        attn_mask_annealing_end_steps=0,        
        lr=1e-4,
        llrd=0.8,
        llrd_l2_enabled=False,                  
        lr_mult=1.0,                            
        weight_decay=0.05,
        poly_power=0.9,                         
        warmup_steps=0,                         
        ckpt_path=ckpt_path,
        delta_weights=False,
        load_ckpt_class_head=True,
    )

    model = lm.network
    model.to(device)
    model.eval()  # Switch to evaluation mode: disables Dropout and freezes Batch Normalization statistics
    return model

def temperature_scaling(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Apply Temperature Scaling to adjust the entropy of the output distribution.
    """
    if temperature <= 0:
        raise ValueError(f"Temperature must be > 0, received: {temperature}")
    return logits / temperature

def compute_anomaly_score(logits: torch.Tensor, method: str) -> torch.Tensor:
    """
    Generate pixel-wise anomaly score map based on the selected scoring function.
    """
    probs = F.softmax(logits, dim=0)

    if method == "msp":
        msp = probs.max(dim=0).values
        anomaly_score = 1.0 - msp

    elif method == "maxentropy":
        eps = 1e-8
        entropy = -(probs * (probs + eps).log()).sum(dim=0)
        anomaly_score = entropy

    elif method == "rba":
        anomaly_score = -logits.tanh().sum(dim=0)

    elif method == "maxlogit":
        maxlogit = logits.max(dim=0).values
        anomaly_score = -maxlogit

    else:
        raise ValueError(f"Anomaly method does not exist: {method}")

    return anomaly_score

# -----------------------------------------------------------------------------
# MAIN LOOP
# -----------------------------------------------------------------------------
def main():
    parser = ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True, help="Path o glob pattern immagini input")
    parser.add_argument("--loadDir", default="", help="Cartella dei modelli salvati")
    parser.add_argument("--loadWeights", default="", help="Nome file checkpoint")
    parser.add_argument("--method", default="msp", choices=["msp", "maxlogit", "maxentropy", "rba"], help="Metodo score anomalia")
    parser.add_argument("--temperature", type=float, default=1.0, help="Valore Temperature Scaling")
    parser.add_argument("--cpu", action="store_true", help="Forza esecuzione su CPU")
    
    # save logits
    parser.add_argument("--save_logits", action="store_true", help="Se attivo, salva i logits su disco e ESCE senza calcolare metriche")
    parser.add_argument("--use_cached", action="store_true", help="Se attivo, carica i logits dal disco invece di usare il modello")
    parser.add_argument("--cache_dir", default="logits_cache", help="Cartella dove salvare/caricare i logits")

    args = parser.parse_args()

    # Device configuration (CPU/GPU)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Device in use: {device}")

    # Create results file if it does not exist
    try:
        if not os.path.exists("results.txt"):
            open("results.txt", "w").close()
    except:
        pass
    
    # -------------------------------------------------------------------------
    # 1. MODEL LOADING (Only if NOT using cached logits)
    # -------------------------------------------------------------------------
    model = None
    if not args.use_cached:
        ckpt_full_path = os.path.join(args.loadDir, args.loadWeights)
        print(f"Loading weights from: {ckpt_full_path}")
        try:
            model = load_eomt(ckpt_full_path, device)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Critical error loading model: {e}")
            return
    else:
        print("CACHE MODE ACTIVE: Neural model will not be loaded (saving GPU/Time).")

    # Create cache directory if in saving mode
    if args.save_logits:
        os.makedirs(args.cache_dir, exist_ok=True)
        print(f"--> Logits will be saved in: {args.cache_dir}")

    # Lists to accumulate results for global metric calculation
    anomaly_score_list = []
    ood_gts_list = []
    
    # Glob pattern handling
    if len(args.input) == 1 and '*' in args.input[0]:
        file_list = glob.glob(args.input[0])
    else:
        file_list = []
        for inp in args.input:
            file_list.extend(glob.glob(inp))
            
    file_list = sorted(file_list)
    print(f"Found {len(file_list)} images to process.")

    # -------------------------------------------------------------------------
    #  elaborating batch
    # -------------------------------------------------------------------------
    for path in file_list:
        
        # Setup cache paths
        filename = os.path.basename(path)
        filename_no_ext = os.path.splitext(filename)[0]
        cache_path = os.path.join(args.cache_dir, filename_no_ext + ".npy")
        pixel_logits = None

        # --- CASE A: INFERENCE & SAVING (Stage 1) - Compute and save logits ---
        if args.save_logits:
            if os.path.exists(cache_path):
                print(f"Cache already exists for {filename}, skipping.")
                continue
            
            print(f"Generating Logits: {filename}")
            img_pil = Image.open(path).convert("RGB")
            img_tensor = input_transform(img_pil).unsqueeze(0).float().to(device)

            with torch.no_grad():
                mask_logits_layers, class_logits_layers = model(img_tensor)
                final_mask_logits = mask_logits_layers[-1]
                final_class_logits = class_logits_layers[-1]

                # Upsample
                final_mask_logits = F.interpolate(
                    final_mask_logits, size=IMG_SIZE, mode="bilinear", align_corners=False
                )

                # Convert mask/class predictions to per-pixel semantic logits [C, H, W]
                per_pixel_logits = LightningModule.to_per_pixel_logits_semantic(
                    final_mask_logits, final_class_logits
                )
                pixel_logits = per_pixel_logits[0]
                
                # Save to disk using float16 to reduce storage requirements
                np.save(cache_path, pixel_logits.cpu().numpy().astype(np.float16))
            
            # Continue to next image in saving mode
            continue

        # --- CASE B: LOADING FROM CACHE (Stage 2) - Fast evaluation ---
        elif args.use_cached:
            if not os.path.exists(cache_path):
                print(f"ERROR: Cache file not found: {cache_path}")
                continue
            
            # Fast loading of pre-computed logits
            loaded_logits = np.load(cache_path).astype(np.float32)
            pixel_logits = torch.from_numpy(loaded_logits).to(device)

        # --- CASE C: STANDARD INFERENCE (No Cache) - Direct evaluation ---
        else:
            print(f"Processing: {path}")
            img_pil = Image.open(path).convert("RGB")
            img_tensor = input_transform(img_pil).unsqueeze(0).float().to(device)

            with torch.no_grad():
                mask_logits_layers, class_logits_layers = model(img_tensor)
                final_mask_logits = mask_logits_layers[-1]
                final_class_logits = class_logits_layers[-1]

                final_mask_logits = F.interpolate(
                    final_mask_logits, size=IMG_SIZE, mode="bilinear", align_corners=False
                )
                per_pixel_logits = LightningModule.to_per_pixel_logits_semantic(
                    final_mask_logits, final_class_logits
                )
                pixel_logits = per_pixel_logits[0]

        # ---------------------------------------------------------------------
        # ANOMALY SCORE COMPUTATION AND METRICS (Common to B and C)
        # ---------------------------------------------------------------------
        
        # 1. Apply Temperature Scaling
        if args.temperature != 1.0:
            pixel_logits = temperature_scaling(pixel_logits, args.temperature)

        # 2. Compute Anomaly Map
        anomaly_score = compute_anomaly_score(pixel_logits, args.method)
        anomaly_np = anomaly_score.detach().cpu().numpy()

        # 3. Load Ground Truth
        pathGT = path.replace("images", "labels_masks")
        if "RoadObsticle21" in pathGT: pathGT = pathGT.replace("webp", "png")
        if "fs_static" in pathGT: pathGT = pathGT.replace("jpg", "png")
        if "RoadAnomaly" in pathGT: pathGT = pathGT.replace("jpg", "png")

        try:
            gt_img = Image.open(pathGT)
        except FileNotFoundError:
            # Fallback for file extensions if exact match is not found
            base_gt = os.path.splitext(pathGT)[0]
            if os.path.exists(base_gt + ".png"): gt_img = Image.open(base_gt + ".png")
            elif os.path.exists(base_gt + ".jpg"): gt_img = Image.open(base_gt + ".jpg")
            else:
                print(f"Warning: GT not found for {path}, skipping.")
                continue

        gt_img = target_transform(gt_img)
        ood_gts = np.array(gt_img)

        # Dataset Label Mapping: Standardize anomaly (1), in-distribution (0), and void (255) classes
        if "RoadAnomaly" in pathGT:
            ood_gts = np.where((ood_gts == 2), 1, ood_gts)
        elif "LostAndFound" in pathGT:
            ood_gts = np.where((ood_gts == 0), 255, ood_gts)
            ood_gts = np.where((ood_gts == 1), 0, ood_gts)
            ood_gts = np.where((ood_gts > 1) & (ood_gts < 201), 1, ood_gts)
        elif "Streethazard" in pathGT:
            ood_gts = np.where((ood_gts == 14), 255, ood_gts)
            ood_gts = np.where((ood_gts < 20), 0, ood_gts)
            ood_gts = np.where((ood_gts == 255), 1, ood_gts)

        # Skip image if no valid anomaly pixels are present
        if 1 not in np.unique(ood_gts):
            continue

        ood_gts_list.append(ood_gts)
        anomaly_score_list.append(anomaly_np)

        # Clear GPU memory (crucial when not using cache to avoid OOM)
        del pixel_logits, anomaly_score
        if not args.use_cached:
            torch.cuda.empty_cache()

    # If saving logits, exit after processing
    if args.save_logits:
        print("Cache Saving Completed.")
        return

    # -------------------------------------------------------------------------
    # FINAL METRICS CALCULATION
    # -------------------------------------------------------------------------
    if not ood_gts_list:
        print("No valid data collected for evaluation.")
        return

    print("Calculating global metrics...")
    
    ood_gts_flat = np.array(ood_gts_list)
    anomaly_scores_flat = np.array(anomaly_score_list)

    ood_mask = (ood_gts_flat == 1)
    ind_mask = (ood_gts_flat == 0)

    ood_scores = anomaly_scores_flat[ood_mask]
    ind_scores = anomaly_scores_flat[ind_mask]

    all_scores = np.concatenate((ind_scores, ood_scores))
    all_labels = np.concatenate((np.zeros(len(ind_scores)), np.ones(len(ood_scores))))

    auprc = average_precision_score(all_labels, all_scores)
    fpr95 = fpr_at_95_tpr(all_scores, all_labels)

    result_str = f"[EoMT-{args.method} | T={args.temperature}] AUPRC: {auprc * 100.0:.2f}% | FPR@95TPR: {fpr95 * 100.0:.2f}%"
    print("\n" + "="*60)
    print(result_str)
    print("="*60 + "\n")

    # Append results to file
    with open("results.txt", "a") as f:
        f.write("\n" + result_str)

if __name__ == "__main__":
    main()
