# Pose Estimation Pipeline for Surgical Tools

This repository implements a modular deep learning pipeline for 6D pose estimation of surgical tools using a modified PoseCNN architecture. The system is built for reproducibility, clear validation, and visual transparency. It is designed for applications in robotic-assisted surgery, where precise localisation of instruments is critical.

---

## 🧠 Project Overview

The goal is to estimate the 6D pose — 3D translation and 3D rotation — of surgical tools from monocular RGB images. This pipeline:

- Accepts RGB + binary mask images as input
- Converts 3x4 pose matrices into quaternion-translation pairs
- Trains a CNN with dual regression heads
- Evaluates performance using geometric and visual metrics
- Validates annotations and rotation representations

It is tailored for datasets with annotated pose matrices, camera intrinsics, and object models.

---

## 🗂 Directory Structure

```bash
pose_estimation_project/
├── data/                    # Dataset and loading utilities
│   ├── dataloader.py        # Custom PyTorch Dataset class
│   └── utils.py             # Preprocessing, augmentation (if added)
├── model/                   # Model definition and losses
│   ├── posecnn.py           # Main CNN model
│   └── loss_functions.py    # Quaternion loss + translation MSE
├── training/                # Training and evaluation logic
│   ├── train.py             # Train loop
│   └── metrics.py           # ADD, ADD-S, 2D Projection error
├── visualisation/           # Drawing and debugging tools
│   ├── draw_pose.py         # Overlay pose axes and keypoints
│   └── vis_utils.py         # (Optional) utils for mask/contour visualisation
├── validation/              # Standalone integrity checkers
│   ├── verify_pose_integrity.py     # Checks .npy matrices
│   └── verify_annotations_json.py   # Confirms unit quaternion annotations
├── utils/                   # Output manager and helper functions
│   └── output_manager.py
├── outputs/                 # Saved logs, models, and figures
├── run.py                   # Unified launcher script
├── requirements.txt         # Dependency list
└── README.md                # This file
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/uol-feps-soc-comp3931-project-2425/individual-project-HassanKh17.git
cd individual-project-HassanKh17
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
You may use a virtual environment for isolation.

### 3. Prepare Your Dataset
Structure your training data as follows:
```
MFB_TRAIN/TRAIN/
├── image_resized/             # RGB images
├── mask/                      # Binary masks
├── pose/                      # .npy pose files (3x4 matrices)
├── posecnn_annotations.json   # Converted JSON with translation + quaternion
├── model_keypoints.npy        # 3D object keypoints
├── camera_matrix.npy          # Intrinsic matrix (3x3)
```

### 4. Launch Training + Evaluation
```bash
python run.py
```
This executes:
- Model training on annotated RGB+mask images
- Visualisation of predicted pose overlays
- Evaluation using ADD, ADD-S, 2D projection error
- Saves results under `outputs/` automatically

---

## 🧪 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **ADD**      | Mean distance between model points transformed by predicted and GT pose |
| **ADD-S**    | ADD with symmetry-aware closest point matching |
| **2D Projection Error** | Average pixel distance of keypoint projections |

Additional metrics or plots (e.g., quaternion norm histograms) are supported via validation scripts.

---

## 🧹 Validation Tools

Standalone scripts for input integrity:

```bash
# Check pose matrices (rank + orthogonality)
python validation/verify_pose_integrity.py --dir MFB_TRAIN/TRAIN/pose

# Check that quaternions are unit vectors
python validation/verify_annotations_json.py --json MFB_TRAIN/TRAIN/posecnn_annotations.json
```

Outputs: pass/fail logs for traceability.

---

## 🖼 Visualisations

Pose overlays are saved to:
```
outputs/<experiment_name>/visualisations/
```
Includes:
- RGB + axis overlay
- Projected keypoints
- Side-by-side examples

---

## 🧾 Logging and Output
All results are automatically structured:
```
outputs/
└── posecnn_experiment_YYYYMMDD_HHMM/
    ├── checkpoints/
    ├── logs/
    │   └── training_log.jsonl
    ├── visualisations/
    └── evaluation_results.json
```
Use `OutputManager` in `utils/` for flexible saving.

---

## ⚙️ Configuration
Modify `run.py` to change:
- Dataset paths
- Object diameter
- Camera intrinsics
- Experiment parameters

---


