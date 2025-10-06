#!/bin/bash

# Download DocTr pretrained models from Hugging Face
# Source: https://huggingface.co/spaces/HaoFeng2019/DocTr/tree/main/model_pretrained

set -e

MODEL_DIR="/oomol-driver/oomol-storage/model_pretrained"
BASE_URL="https://huggingface.co/spaces/HaoFeng2019/DocTr/resolve/main/model_pretrained"

echo "Creating model directory at $MODEL_DIR..."
mkdir -p "$MODEL_DIR"

echo "Downloading DocTr pretrained models..."

# Download seg.pth (4.5MB)
if [ ! -f "$MODEL_DIR/seg.pth" ] || [ ! -s "$MODEL_DIR/seg.pth" ]; then
    echo "Downloading seg.pth (4.5MB)..."
    wget -q --show-progress -O "$MODEL_DIR/seg.pth" \
        "$BASE_URL/seg.pth"
else
    echo "seg.pth already exists, skipping..."
fi

# Download geotr.pth (99MB)
if [ ! -f "$MODEL_DIR/geotr.pth" ] || [ ! -s "$MODEL_DIR/geotr.pth" ]; then
    echo "Downloading geotr.pth (99MB)..."
    wget -q --show-progress -O "$MODEL_DIR/geotr.pth" \
        "$BASE_URL/geotr.pth"
else
    echo "geotr.pth already exists, skipping..."
fi

# Download illtr.pth (50MB)
if [ ! -f "$MODEL_DIR/illtr.pth" ] || [ ! -s "$MODEL_DIR/illtr.pth" ]; then
    echo "Downloading illtr.pth (50MB)..."
    wget -q --show-progress -O "$MODEL_DIR/illtr.pth" \
        "$BASE_URL/illtr.pth"
else
    echo "illtr.pth already exists, skipping..."
fi

echo "✓ All models downloaded successfully to $MODEL_DIR"
echo "  - seg.pth (4.5MB) - Document segmentation model"
echo "  - geotr.pth (99MB) - Geometric transformation model"
echo "  - illtr.pth (50MB) - Illumination correction model"
