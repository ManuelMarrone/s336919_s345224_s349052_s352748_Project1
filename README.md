# Mask Architecture Anomaly Segmentation for Road Scenes

This repository explores various methodologies for anomaly detection in urban environments, ranging from traditional pixel-based post-hoc methods to advanced mask-based architectures and novel **Outlier Exposure (OE)** training strategies.

## 1. Pixel-based Baselines

The initial phase of this research focuses on conventional pixel-level anomaly detection using post-hoc uncertainty estimation methods. We utilize an **ERFNet** model pretrained on Cityscapes to evaluate baseline performance on standard anomaly segmentation validation sets.

* **Implementation**: `eval/eval_iou.py` and `eval/evalAnomaly.py`
* **Methodology**: We compute anomaly scores by extracting uncertainty maps from the model's output distribution.
* **Evaluated Methods**:
    * **MSP (Maximum Softmax Probability)**: The baseline probability distribution.
    * **Max Logit**: Utilizing the raw output values before normalization.
    * **Max Entropy**: Measuring the information-theoretic uncertainty of the pixel-wise predictions.



## 2. Mask-based Baselines & Mask-specific Metrics

This section transitions from pixel-wise classification to an object-centric approach using the **EoMT (Evidence-based Object-centric Mask Transformer)** architecture.

* **Implementation**: `eomt/eval_iou_eomt.py` and `eomt/temperature_scaling/eval_anomaly_temp_eomt.py`
* **Key Shift**: Unlike pixel-based models, mask architectures output discrete segments. We adapt the inference pipeline to handle these structures while implementing specialized anomaly scores:
    * **MSP, Max Logit, and Max Entropy**: Adapted for mask-level predictions.
    * **RbA (Reliability by Answering)**: A mask-specific scoring method that leverages the architecture's ability to "question" the validity of a detected object.



## 3. Confidence Calibration via Temperature Scaling

To further optimize anomaly detection performance, we implement **Temperature Scaling** as a post-hoc calibration technique. This method aims to align the model’s confidence with its actual predictive accuracy.

* **Implementation**: `eomt/temperature_scaling/`

## 4. RbA with Outlier Exposure (Baseline OE)

Following the methodology proposed in the **RbA paper**, we implement an Outlier Exposure (OE) framework to regularize the model's behavior on out-of-distribution (OOD) samples.

* **Implementation**: `RbAMethodOE/OE_method.ipynb`
* **Approach**: This module integrates external outliers during the training or refinement phase. By forcing the model to assign lower confidence (or higher entropy) to these known outliers, we improve the discriminative boundary between "road" objects and anomalies. This implementation serves as a benchmark for our custom OE strategies.

## 5. Proposed Outlier Exposure Pipeline

The final module introduces our custom **OE training pipeline**, designed to enhance the robustness of the segmentation model through synthetic data augmentation.

* **Implementation**: `our_OE_Method/`
* **Technical Strategy**:
    * **Data Augmentation**: We employ a **Cut-and-Paste** strategy, where objects from the **COCO dataset** (non-urban classes) are synthetically injected into **Cityscapes** frames.
    * **Objective**: The pipeline trains the model to recognize these foreign objects as "anomalous" while maintaining high mIoU on the primary task.
