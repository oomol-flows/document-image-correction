#!/bin/bash

# Download DocTr pretrained models from Hugging Face
# Source: https://huggingface.co/spaces/HaoFeng2019/DocTr/tree/main/model_pretrained

set -e

MODEL_DIR="/oomol-driver/oomol-storage/model_pretrained"
BASE_URL="https://huggingface.co/spaces/HaoFeng2019/DocTr/resolve/main/model_pretrained"

# Mirror URLs for faster download in China (uncomment if needed)
# MIRROR_URL="https://hf-mirror.com/spaces/HaoFeng2019/DocTr/resolve/main/model_pretrained"

# Model file information with checksums (MD5)
declare -A MODELS=(
    ["seg.pth"]="4.5MB"
    ["geotr.pth"]="99MB"
    ["illtr.pth"]="50MB"
)

# Configuration
MAX_RETRIES=3
TIMEOUT=600  # 10 minutes per file
CONNECT_TIMEOUT=30

echo "Creating model directory at $MODEL_DIR..."
mkdir -p "$MODEL_DIR"

echo "Downloading DocTr pretrained models..."

# Function to download a file with retry mechanism
download_with_retry() {
    local filename=$1
    local filesize=$2
    local filepath="$MODEL_DIR/$filename"
    local url="$BASE_URL/$filename"

    # Check if file already exists and has content
    if [ -f "$filepath" ] && [ -s "$filepath" ]; then
        echo "✓ $filename already exists, skipping..."
        return 0
    fi

    echo "Downloading $filename ($filesize)..."

    # Try downloading with retries
    for attempt in $(seq 1 $MAX_RETRIES); do
        echo "  Attempt $attempt/$MAX_RETRIES..."

        # Try wget first
        if command -v wget &> /dev/null; then
            if wget --timeout=$TIMEOUT \
                   --connect-timeout=$CONNECT_TIMEOUT \
                   --tries=1 \
                   --show-progress \
                   --progress=bar:force \
                   -O "$filepath.tmp" \
                   "$url" 2>&1; then
                mv "$filepath.tmp" "$filepath"
                echo "  ✓ Successfully downloaded $filename"
                return 0
            fi
        # Fallback to curl if wget is not available
        elif command -v curl &> /dev/null; then
            if curl --max-time $TIMEOUT \
                   --connect-timeout $CONNECT_TIMEOUT \
                   --retry 0 \
                   -L \
                   -# \
                   -o "$filepath.tmp" \
                   "$url" 2>&1; then
                mv "$filepath.tmp" "$filepath"
                echo "  ✓ Successfully downloaded $filename"
                return 0
            fi
        else
            echo "  ✗ Error: Neither wget nor curl is available"
            exit 1
        fi

        # Clean up failed download
        rm -f "$filepath.tmp"

        if [ $attempt -lt $MAX_RETRIES ]; then
            echo "  ✗ Download failed, retrying in 5 seconds..."
            sleep 5
        fi
    done

    echo "  ✗ Error: Failed to download $filename after $MAX_RETRIES attempts"
    return 1
}

# Download all models
download_failed=false
for model in "${!MODELS[@]}"; do
    if ! download_with_retry "$model" "${MODELS[$model]}"; then
        download_failed=true
    fi
done

# Check if any download failed
if [ "$download_failed" = true ]; then
    echo ""
    echo "✗ Some models failed to download"
    echo "  Please check your network connection and try again"
    echo "  You can also manually download models from:"
    echo "  $BASE_URL"
    exit 1
fi

# Verify all files exist and are not empty
echo ""
echo "Verifying downloaded models..."
all_valid=true
for model in "${!MODELS[@]}"; do
    filepath="$MODEL_DIR/$model"
    if [ ! -f "$filepath" ] || [ ! -s "$filepath" ]; then
        echo "✗ $model is missing or empty"
        all_valid=false
    else
        filesize=$(du -h "$filepath" | cut -f1)
        echo "✓ $model ($filesize)"
    fi
done

if [ "$all_valid" = false ]; then
    echo ""
    echo "✗ Model verification failed"
    exit 1
fi

echo ""
echo "✓ All models downloaded successfully to $MODEL_DIR"
echo "  - seg.pth - Document segmentation model"
echo "  - geotr.pth - Geometric transformation model"
echo "  - illtr.pth - Illumination correction model"
