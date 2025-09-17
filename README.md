# SAR-RARP50: A Detection-Segmentation Approach

This project explores methods for surgical device segmentation using the **SAR-RARP50** dataset ([Synapse link](https://www.synapse.org/Synapse:syn27618412/wiki/617968)). This dataset contains videos of **Robot-Assisted Radical Prostatectomy** (RARP), with segmentation masks of surgical instruments available every 60 video frames. The task combines **detection, segmentation, and video coherence**.
The approach is based on the **YOLO11n** model for live tracking ([Ultralytics YOLO](https://github.com/ultralytics/ultralytics)). The main focus of this work is the **combination of YOLO with SAM2**, two foundational models that show strong performance on natural images for tracking, detection, and segmentation. The idea is to use YOLO to obtain tracking bounding boxes and then prompt **SAM2** for segmentation.
All the checkpoints are available in this repository: [Google Drive link](https://drive.google.com/drive/u/0/folders/1tpXC1PhkBDdYjXW6sGeS4Ko2BhKJqITl)


# Methodology


#### 1. Data Preparation
The first step is to prepare the dataset for **video object tracking and detection**. To achieve this, segmentation masks are separated using **connected component-based analysis**, and bounding boxes are fitted at the maximum and minimum coordinates of each component. The frames are then downsampled to a resolution of **426×240** for faster training and inference. This preprocessing is implemented in `yolo_sam/dataset/save_datasets.py`, and the extracted bounding boxes can be visualized in `yolo_sam/dataset/viewer.ipynb`. The data must be converted into the appropriate formats for SAM2 and YOLO training. 

#### 2.1. Fine-tuning YOLO11n
SAR-RARP50-train dataset is split into a 80:20 train:val split. The YOLO11n model is **fine-tuned from the COCO checkpoint** on all available annotated frames in the dataset. This allows the model to adapt to the surgical instrument domain while leveraging the pre-trained knowledge from natural images. The model was trained for 100 epochs (approx 2h).

#### 2.2 Fine-tuning YOLO11n
For comparison purposes, a **YOLO11n-seg** model is also trained, which directly outputs both bounding boxes and segmentation masks, without relying on SAM2. This allows assessment of the benefits of the YOLO+SAM2 pipeline versus an end-to-end segmentation approach. The model was trained for 100 epochs (approx 2h).

#### 2.3 Fine-tuning SAM2.1
The **SAM2.1 image encoder** is trained on SAR-RARP50 using **Low Rank Adaptation (LoRA)**. During training, a high probability of prompting with bounding boxes is used (`prob_to_use_box_input_for_train: 0.7` in `yolo_sam/sam2/sam2/configs/sam2.1_training/sam2.1_hiera_t_lora_sar_rarp50.yaml`) to improve the model's ability to segment surgical instruments given box. It is worth noting that SAM2 is trained in a "instance-segmentation" meaning that objects appearing on the 2 different sides of the images are quite hard to segment with a single box. For this purpose, some classes (especially the three "Tool" classes) were subdived into multiple objects. The model was trained for 9 epochs (approx 2h30).

#### 3. Model Predictions
For inference, the **fine-tuned YOLO model** is run on each frame of videos from the test-set. The detected bounding box coordinates are then used to **prompt the fine-tuned SAM2 model**, which produces the final segmentation predictions.  

#### 4. Evaluation
The models are evaluated using several metrics. Frame-wise performance is assessed using the Dice coefficient and Intersection over Union (IoU) for each class. Video-level performance is evaluated using mean IoU (mIoU) and Normalized Surface Dice (NSD) to assess overall performance across videos. The evaluation scripts are located in `yolo_sam/evaluation/evaluations`, and visual inspections of model performance can be explored in the notebook `visual_evaluation.ipynb`.  
This frame-by-frame approach was also chosen with the perspective of real-time applications in mind. For this reason, a short inference-time analysis was additionally conducted in the notebook `visual_evaluation.ipynb`.

---

## Installation
Instructions to set up the project locally.

```bash
# Clone the repository
git clone https://github.com/username/repo-name.git

# Install dependencies
pip install -r requirements.txt
```

---
## Usage

Download-Unzip "SAR-RARP50_train_set.zip" and "SAR-RARP50_test_set.zip":
```bash
python .../data/download_data.py
```

Convert the dataset into the right format:
```bash
python .../yolo_sam/dataset/save_datasets.py
```

Train YOLO11n for box detection:
```bash
bash .../yolo_sam/yolo/train_yolo_det.sh
```

Train YOLO11n-seg for segmentation:
```bash
bash .../yolo_sam/yolo/train_yolo_seg.sh
```

Train SAM2 Image encoder for segmentation:
```bash
bash .../yolo_sam/sam2/script.sh
```

For YOLO and SAM2 predictions run:
```bash
python .../yolo_sam/evaluation/predictions/sam_predictions.py
python .../yolo_sam/evaluation/predictions/yolo-seg_predictions.py
```

