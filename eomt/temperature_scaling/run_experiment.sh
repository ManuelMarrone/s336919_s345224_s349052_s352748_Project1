#!/bin/bash

# ==============================================================================
# CONFIG PATH
# ==============================================================================
SCRIPT_PATH="/content/drive/MyDrive/temperature/MaskArchitectureAnomaly_CourseProject-main/MaskArchitectureAnomaly_CourseProject-main/eomt/eval_anomaly_temp_eomt.py"
WEIGHTS_DIR="/content/drive/MyDrive/temperature/MaskArchitectureAnomaly_CourseProject-main/MaskArchitectureAnomaly_CourseProject-main/eomt/checkpoints"
WEIGHTS_FILE="eomt_cityscapes.bin"
BASE_DATA_DIR="/content/drive/MyDrive/temperature/MaskArchitectureAnomaly_CourseProject-main/MaskArchitectureAnomaly_CourseProject-main/datasets/Validation_Dataset"
CACHE_ROOT="/content/drive/MyDrive/temperature/MaskArchitectureAnomaly_CourseProject-main/MaskArchitectureAnomaly_CourseProject-main/logits_cache"

# ==============================================================================
# PARAMETERS
# ==============================================================================
DATASETS=("FS_LostFound_full" "fs_static" "RoadAnomaly" "RoadAnomaly21" "RoadObsticle21")
METHODS=("msp" "rba" "maxlogit" "maxentropy")
TEMPERATURES=(0.5 0.75 1.0 1.2 1.4 1.6 1.8 2.0 2.2 2.4 2.6 2.8 3.0 5.0 100.0)

# Move to the script directory to handle imports
cd "$(dirname "$SCRIPT_PATH")" || exit

# ==============================================================================
# PHASE 1: INFERENCE & CACHING (Executed once per Dataset)
# ==============================================================================
echo "----------------------------------------------------------------"
echo "PHASE 1: LOGITS CACHE GENERATION (Heavy Computation)"
echo "----------------------------------------------------------------"

for dataset in "${DATASETS[@]}"; do
    
    # Setup extensions
    case "$dataset" in
        "FS_LostFound_full") EXT="png" ;;
        "fs_static")         EXT="jpg" ;;
        "RoadAnomaly")       EXT="jpg" ;;
        "RoadAnomaly21")     EXT="png" ;;
        "RoadObsticle21")    EXT="webp" ;;
        *)                   EXT="png" ;;
    esac

    INPUT_PATH="$BASE_DATA_DIR/$dataset/images/*.$EXT"
    CURRENT_CACHE="$CACHE_ROOT/$dataset"
    
    echo ">> Caching dataset: $dataset"
    
    # Run with --save_logits
    python -W ignore "$SCRIPT_PATH" \
        --input "$INPUT_PATH" \
        --loadDir "$WEIGHTS_DIR" \
        --loadWeights "$WEIGHTS_FILE" \
        --save_logits \
        --cache_dir "$CURRENT_CACHE"
done

# ==============================================================================
# PHASE 2: QUICK EVALUATION (Executed NxMxK times)
# ==============================================================================
echo " "
echo "----------------------------------------------------------------"
echo "PHASE 2: METRICS AND TEMPERATURES EVALUATION (Light Computation)"
echo "----------------------------------------------------------------"

for dataset in "${DATASETS[@]}"; do
    
    # Setup extensions
    case "$dataset" in
        "FS_LostFound_full") EXT="png" ;;
        "fs_static")         EXT="jpg" ;;
        "RoadAnomaly")       EXT="jpg" ;;
        "RoadAnomaly21")     EXT="png" ;;
        "RoadObsticle21")    EXT="webp" ;;
        *)                   EXT="png" ;;
    esac

    INPUT_PATH="$BASE_DATA_DIR/$dataset/images/*.$EXT"
    CURRENT_CACHE="$CACHE_ROOT/$dataset"

    echo "================================================================"
    echo "DATASET: $dataset"
    echo "================================================================"

    for method in "${METHODS[@]}"; do
        for temp in "${TEMPERATURES[@]}"; do
            
            echo "Running: Method=$method | Temp=$temp"
            
            # Run with --use_cached (NO MODEL LOADING!)
            python -W ignore "$SCRIPT_PATH" \
                --input "$INPUT_PATH" \
                --method "$method" \
                --temperature "$temp" \
                --use_cached \
                --cache_dir "$CURRENT_CACHE"
        done
    done
done
