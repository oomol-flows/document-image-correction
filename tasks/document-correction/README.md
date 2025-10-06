# Document Distortion Correction

This OOMOL task block corrects distorted document images using deep learning-based geometric transformation.

## Features

- **Automatic Document Segmentation**: Uses U2NETP model to extract document regions from background
- **Geometric Correction**: Applies GeoTr transformer-based model to unwarp distorted documents
- **End-to-End Processing**: Single-step correction from distorted to flattened document

## Model Architecture

The task combines two neural networks:

1. **U2NETP**: Lightweight segmentation model that identifies document boundaries
2. **GeoTr**: Geometric transformer that predicts backward mapping flow for perspective correction

## Usage

### Input

- `input_image`: Path to distorted document image (JPG, PNG, BMP, TIFF)

### Output

- `output_image`: Path to corrected document image

### Example

The task automatically:
1. Loads the input image
2. Segments the document region
3. Applies geometric correction
4. Saves the corrected image to `/oomol-driver/oomol-storage/`

## Technical Details

- **Model Size**: ~103 MB (seg.pth: 4.5MB, geotr.pth: 99MB)
- **Input Resolution**: Images are resized to 288x288 for processing, then upsampled back to original size
- **Framework**: PyTorch
- **Dependencies**: torch, torchvision, opencv-python-headless, pillow, numpy, scikit-image, timm

## Credits

Based on the DocTr (Document Transformer) research project for document image rectification.
