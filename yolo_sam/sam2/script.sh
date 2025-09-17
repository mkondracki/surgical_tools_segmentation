#!/bin/bash

cd /data/mkondrac/foundation_model_cardio/code/SAM/yolo_sam/sam2 || { echo "Failed to change directory"; exit 1; }

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=$PYTHONPATH:$(pwd)

PYTHON_ENV=/data/mkondrac/miniconda3/envs/sam/bin/python
TRAIN_SCRIPT=/data/mkondrac/foundation_model_cardio/code/SAM/yolo_sam/sam2/training/train.py

CONFIG=configs/sam2.1_training/sam2.1_hiera_t_lora_sar_rarp50.yaml

for i in {1..30}; do
    echo "Attempt $i at $(date)"
    $PYTHON_ENV $TRAIN_SCRIPT -c "$CONFIG" --use-cluster 0 --num-gpus 1 && break
    echo "Attempt $i failed at $(date)"
done

echo "Training script finished at $(date)"