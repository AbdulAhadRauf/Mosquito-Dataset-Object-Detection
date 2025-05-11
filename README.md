# DLP Object Detection Week 10 – Mosquito Dataset Object Detection

## 📄 Competition Link

[DLP Object Detection Week 10 on Kaggle](https://www.kaggle.com/competitions/dlp-object-detection-week-10/overview)

## 🏆 Overview

Automate the detection and classification of mosquitoes in real-world images using deep learning object detection techniques. Build a model that locates mosquitoes via bounding boxes and classifies each into one of six species.

* **Task:** Object detection & classification of mosquitoes
* **Start:** March 14, 2025
* **Close:** March 26, 2025
* **Metric:** Mean Average Precision (mAP)

## 🗃️ Dataset Description

* **Total Images:** 8,025 real-world mosquito photographs
* **Train/Test Split:** 7,500 images (93.5%) for training, 525 images (6.5%) for testing
* **Annotations:** YOLO-format `.txt` files per image with:

  * `class_label` (0–5)
  * `bbx_xcenter`, `bbx_ycenter`, `bbx_width`, `bbx_height` (normalized)
* **Classes:**

  1. `aegypti` (0)
  2. `albopictus` (1)
  3. `anopheles` (2)
  4. `culex` (3)
  5. `culiseta` (4)
  6. `japonicus/koreicus` (5)

### Directory Structure

```
├── data/
│   ├── train/
│   │   ├── images/    # 7,500 training images
│   │   └── labels/    # Corresponding YOLO-format annotations
│   └── test/
│       ├── images/    # 525 test images
│       └── labels/    # (not provided)
├── notebooks/         # EDA, augmentation, training notebooks
├── src/               # Custom modules (datasets, utils)
├── outputs/           # Checkpoints, logs, inference results
├── submission.py      # Generates `submission.csv` from predictions
└── README_ObjDet_Week10.md  # This file
```

## 🛠️ Environment & Dependencies

* **Python 3.8+**
* **PyTorch** & **Torchvision**
* **YOLOv5** (Ultralytics)
* **Detectron2** (optional)
* **EfficientDet** (TensorFlow/PyTorch impl)
* **OpenCV**, **Pillow**
* **albumentations** (augmentations)
* **NumPy**, **Pandas**
* **Matplotlib**, **Seaborn** (visualization)
* **tqdm** (progress bars)

Install via:

```bash
pip install -r requirements.txt
```

## 🔍 Exploratory Data Analysis (EDA)

1. **Class Distribution:** bar plots of sample counts per mosquito species.
2. **Image Samples:** grid view of annotated images.
3. **Bounding Box Statistics:** distribution of box sizes and aspect ratios.
4. **Data Quality Check:** overlapping boxes, missing labels.

## 🏗️ Model Architectures & Approach

### 1. YOLOv5

* **Framework:** Ultralytics’ YOLOv5 (v6.x)
* **Backbone:** CSPDarknet53
* **Head:** PANet + YOLO detection layers
* **Key Features:** real-time speed, strong accuracy with small objects
* **Training Details:**

  * Transfer learning from COCO pretrained weights
  * Image size: 640×640
  * Batch size: 16
  * Augmentations: mosaic, mixup, random flips, hue/saturation
  * Optimizer: AdamW (lr=0.001)
  * Epochs: 50 with cosine LR schedule

### 2. Faster R-CNN

* **Framework:** Detectron2
* **Backbone:** ResNet-50 + FPN
* **ROI Head:** Two-stage region proposals and classification
* **Key Features:** robust for variable sizes, high detection precision
* **Training Details:**

  * Pretrained on COCO
  * Batch size: 8
  * Learning rate: 0.00025 (step decay)
  * Epochs: 20
  * Anchor scales tuned for small objects

### 3. EfficientDet

* **Architecture:** BiFPN feature fusion and Compound Scaling
* **Variant:** EfficientDet-D2 (balanced accuracy & speed)
* **Key Features:** efficient multi-scale feature aggregation
* **Training Details:**

  * Transfer learning from pretrained checkpoints
  * Input size: 512×512
  * Batch size: 8
  * Augmentations: random crop, brightness/contrast, rotation
  * Optimizer: Adam (lr=0.0003)
  * Epochs: 30

## 🎓 Training Pipeline

1. **Dataset Class:** custom `MosquitoDataset` extends `torch.utils.data.Dataset`.
2. **DataLoader:** shuffling, multi-threaded loading, collate fn for bounding boxes.
3. **Augmentation:** `albumentations` pipeline applied on-the-fly.
4. **Training Loop:** forward pass, compute classification & box regression losses, backward, step.
5. **Validation:** compute mAP at IoU=0.50:0.95 on validation split.
6. **Checkpointing:** save best model by validation mAP.

## 📊 Evaluation

* **mAP@\[.5:.95]:** primary metric for leaderboard ranking.
* **AP per Class:** monitor species-wise performance to detect class imbalance issues.

## 🚀 Inference & Submission

1. **Load Checkpoint:** best weights for chosen model.
2. **Run Inference:** on `data/test/images` with NMS threshold=0.5.
3. **Format Predictions:** convert boxes & labels to sample submission format:

   ```
   image_id,xmin,ymin,xmax,ymax,confidence,class_id
   ```
4. **Generate CSV:** `submission.py` collates all detections into `submission.csv`.

```bash
python submission.py --model yolov5 --weights runs/yolov5/exp/weights/best.pt \
    --source data/test/images --output submissions/yolov5_mosquito.csv
```

## 💡 Skills & Techniques Demonstrated

* **Object Detection Fundamentals:** anchor-based and single-shot detectors
* **PyTorch & Detectron2:** custom datasets, training loops, inference
* **Data Augmentation:** improving robustness to real-world variance
* **Performance Tuning:** hyperparameter search, learning rate schedules
* **Model Comparison:** trade-offs between speed (YOLOv5) and accuracy (Faster R-CNN)
* **Visualization:** bounding box overlays, precision-recall curves
* **Kaggle Workflow:** dataset handling, notebook structuring, submission generation

## 🏃 How to Reproduce

1. Clone this repo and navigate to `notebooks/`.
2. Install dependencies.
3. Place the dataset under `data/`.
4. Run the EDA notebook: `eda_mosquito.ipynb`.
5. Train YOLOv5: `train_yolov5_mosquito.ipynb` or via CLI.
6. Optionally train Faster R-CNN/EfficientDet in respective notebooks.
7. Execute `submission.py` to produce `submission.csv`.

---

**Let’s squish those mosquitoes!**
