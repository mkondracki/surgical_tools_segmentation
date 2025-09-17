#!/bin/sh
# Script to download MedSAM2 model checkpoints
# Create checkpoints directory if it doesn't exist
mkdir -p checkpoints
# Use either wget or curl to download the checkpoints
if command -v wget > /dev/null 2>&1; then
    CMD="wget -P checkpoints"
elif command -v curl > /dev/null 2>&1; then
    CMD="curl -L -o"
    CURL=1
else
    echo "Please install wget or curl to download the checkpoints."
    exit 1
fi
# Define the base URL for MedSAM2 models on Hugging Face
HF_BASE_URL="https://huggingface.co/wanglab/MedSAM2/resolve/main"
# Define the model checkpoint files (as separate variables instead of an array)
MODEL1="MedSAM2_2411.pt"
MODEL2="MedSAM2_US_Heart.pt"
MODEL3="MedSAM2_MRI_LiverLesion.pt"
MODEL4="MedSAM2_CTLesion.pt"
MODEL5="MedSAM2_latest.pt"

# Download each checkpoint
for model in $MODEL1 $MODEL2 $MODEL3 $MODEL4 $MODEL5; do
    echo "Downloading ${model}..."
    model_url="${HF_BASE_URL}/${model}"
    
    if [ -n "$CURL" ]; then
        $CMD "checkpoints/${model}" "$model_url" || { echo "Failed to download checkpoint from $model_url"; exit 1; }
    else
        $CMD "$model_url" || { echo "Failed to download checkpoint from $model_url"; exit 1; }
    fi
done
echo "All MedSAM2 model checkpoints have been downloaded successfully to the 'checkpoints' directory."