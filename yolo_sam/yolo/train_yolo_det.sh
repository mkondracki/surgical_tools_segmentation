
#!/bin/bash

# Dataset path
DATASET_CONFIG="/data/mkondrac/foundation_model_cardio/code/SAM/yolo_sam/yolo/sar-rarp50-det.yaml"

echo "Starting YOLO training..."

# Train YOLOv11n
yolo detect train \
    data=$DATASET_CONFIG \
    model=/data/mkondrac/foundation_model_cardio/code/SAM/yolo_sam/yolo/surgical_tools/yolo11n_surgery/weights/best.pt \
    epochs=100 \
    imgsz=416 \
    batch=256 \
    lr0=0.01 \
    patience=20 \
    device=1 \
    project=surgical_tools_det \
    name=yolo11n_surgery \
    resume=True

echo "Training completed!"