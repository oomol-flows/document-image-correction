#!/bin/bash

# Download DocTr pretrained models from Hugging Face
# Source: https://huggingface.co/spaces/HaoFeng2019/DocTr/tree/main/model_pretrained

set -e

MODEL_DIR="/oomol-driver/oomol-storage/model_pretrained"

echo "Creating model directory at $MODEL_DIR..."
mkdir -p "$MODEL_DIR"

echo "Downloading DocTr pretrained models..."

# Download seg.pth
if [ ! -f "$MODEL_DIR/seg.pth" ] || [ ! -s "$MODEL_DIR/seg.pth" ]; then
    echo "Downloading seg.pth..."
    wget -q --show-progress -O "$MODEL_DIR/seg.pth" \
        "https://huggingface.co/spaces/HaoFeng2019/DocTr/resolve/main/model_pretrained/seg.pth"
else
    echo "seg.pth already exists, skipping..."
fi

# Download geotr.pth
if [ ! -f "$MODEL_DIR/geotr.pth" ] || [ ! -s "$MODEL_DIR/geotr.pth" ]; then
    echo "Downloading geotr.pth..."
    wget -q --show-progress -O "$MODEL_DIR/geotr.pth" \
        "https://huggingface.co/spaces/HaoFeng2019/DocTr/resolve/main/model_pretrained/geotr.pth"
else
    echo "geotr.pth already exists, skipping..."
fi

echo "✓ Models downloaded successfully to $MODEL_DIR"
