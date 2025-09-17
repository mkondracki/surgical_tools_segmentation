
#!/bin/bash

# Dataset path
DATASET_CONFIG="/data/mkondrac/foundation_model_cardio/code/SAM/yolo_sam/yolo/sar-rarp50-seg.yaml"

echo "Starting YOLO training..."

# Train YOLOv11n
yolo detect train \
    data=$DATASET_CONFIG \
    model=yolo11n-seg.pt \
    epochs=100 \
    imgsz=416 \
    batch=256 \
    lr0=0.01 \
    patience=20 \
    device=0 \
    project=surgical_tools_seg \
    name=yolo11n_surgery

echo "Training completed!"