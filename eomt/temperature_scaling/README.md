# Google Colab Execution Guide

This guide describes the steps necessary to configure the environment, organize data, and run anomaly detection experiments using the `run_experiment.sh` script.

---

## 1. Folder Organization (CRUCIAL)
Before starting, it is fundamental that your Google Drive is organized correctly. The script expects to find the datasets in a specific folder.

Ensure that the structure in your Drive looks like this:

```text
/content/drive/MyDrive/YOUR_PROJECT/
├── datasets/                      <-- [IMPORTANT] Create this folder!
│   ├── FS_LostFound_full/         <-- Insert unzipped images here
│   ├── fs_static/
│   ├── RoadAnomaly/
│   ├── RoadAnomaly21/
│   └── RoadObsticle21/
├── eomt/
│   ├── eval_anomaly_eomt.py       <-- The Python script
│   ├── run_experiment.sh          <-- The Bash startup script
│   ├── requirements.txt
│   └── checkpoints/               <-- Folder with weights (.bin)
└── logits_cache/                  <-- (Will be created automatically by the script)
```

## 2. Dependency Installation
Run this cell in your Colab notebook to install the necessary libraries. Make sure you have mounted Google Drive first.
from google.colab import drive
drive.mount('/content/drive')

### Install dependencies
!pip install lightning
# Modify the path below if your requirements.txt is elsewhere
!pip install -r "/content/drive/MyDrive/YOUR_PROJECT/eomt/requirements.txt"

## 3. Path Configuration
BEFORE RUNNING THE SCRIPT: You must open the run_experiment.sh file and modify the variables at the top so they correspond exactly to the location of the files in your Google Drive.

Open run_experiment.sh and modify these lines:

### EXAMPLE OF WHAT TO MODIFY IN run_experiment.sh

# 1. Where the python script is located
SCRIPT_PATH="/content/drive/MyDrive/YOUR_PROJECT/eomt/eval_anomaly_eomt.py"

# 2. Where the model weights are saved
WEIGHTS_DIR="/content/drive/MyDrive/YOUR_PROJECT/eomt/checkpoints"

# 3. [IMPORTANT] The folder containing all datasets
BASE_DATA_DIR="/content/drive/MyDrive/YOUR_PROJECT/datasets"

# 4. Where to save temporary results
CACHE_ROOT="/content/drive/MyDrive/YOUR_PROJECT/logits_cache"

## 4. Running the Experiment
Once the libraries are installed and the paths in the bash file are configured, start the experiment with the following command.

The script will automatically execute:

Inference: Calculation of anomalies and saving of logits (slow, uses GPU).

Evaluation: Calculation of metrics with different temperatures (fast, uses cache).

Command:
# Replace with the correct path to your .sh file
!bash "/content/drive/MyDrive/YOUR_PROJECT/eomt/run_experiment.sh"
