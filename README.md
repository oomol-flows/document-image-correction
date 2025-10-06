# Document Processing Suite

A comprehensive OOMOL workflow toolkit for document image enhancement, featuring automatic correction of distortion, perspective issues, and lighting problems using advanced deep learning technology.

## What Does This Toolkit Do?

This toolkit provides two powerful document processing capabilities:

1. **Geometric Correction**: Automatically fixes warped, curved, or skewed documents - transforming crooked photos into clean, flat, readable images like professional scans.

2. **Illumination Correction**: Fixes uneven lighting, shadows, and poor illumination - making documents captured in bad lighting conditions clear and readable.

### Perfect For:

- **Mobile Phone Photography**: Fix documents photographed at angles, with poor lighting, or in challenging conditions
- **Curved Documents**: Straighten images of books, magazines, or papers that are bent or folded
- **Low-Light Captures**: Enhance documents photographed in dim lighting or with uneven illumination
- **Quick Digitization**: Convert phone-captured documents into professional-looking scans
- **Archive Restoration**: Clean up old or damaged document photographs

## How It Works

These tools use state-of-the-art AI technology powered by Transformer models:

### Geometric Correction
1. **Detect the Document**: Automatically identifies the document region in your image
2. **Analyze Distortion**: Understands how the document is warped or skewed
3. **Apply Correction**: Intelligently transforms the image to flatten and straighten the document
4. **Output Clean Result**: Produces a corrected, readable document image

### Illumination Correction
1. **Analyze Lighting**: Detects uneven illumination, shadows, and lighting issues
2. **Process in Patches**: Divides the image into overlapping patches for detailed correction
3. **Apply Enhancement**: Uses transformer-based deep learning to correct each patch
4. **Reconstruct Image**: Seamlessly combines corrected patches into a uniformly lit document

## Available Blocks

### 1. Document Distortion Correction Block

**What it does**: Corrects distorted, warped, and skewed document images to produce flattened, readable outputs

**Input**:
- Distorted document image (supports JPG, PNG, BMP, TIFF formats)

**Output**:
- Corrected, flattened document image

**Use Cases**:
- Fix perspective distortion from angled photos
- Straighten curved pages from books or folded papers
- Convert mobile phone document captures to professional scans
- Prepare documents for OCR (text recognition)

**Model**: GeoTr (Geometric Transformer) with U2-Net segmentation

---

### 2. Document Illumination Correction Block

**What it does**: Corrects uneven lighting, shadows, and illumination issues in document images

**Input**:
- Document image with poor or uneven lighting (supports JPG, PNG, BMP, TIFF formats)

**Output**:
- Uniformly illuminated, enhanced document image

**Use Cases**:
- Fix documents photographed in poor lighting conditions
- Remove shadows and uneven lighting from document captures
- Enhance visibility of text in dark or poorly lit photos
- Improve document quality before OCR or archiving
- Correct lighting issues from flash photography or desk lamps

**Model**: IllTr (Illumination Transformer)

## Getting Started

### Requirements

- OOMOL Platform installed
- Python 3.10 - 3.12
- No technical knowledge required!

### Quick Start

1. **Open OOMOL Platform**
2. **Load this project**
3. **Create or open a workflow**
4. **Add blocks to your workflow**:
   - For geometric correction: Add "Document Distortion Correction" block
   - For lighting issues: Add "Document Illumination Correction" block
   - For both: Chain both blocks together!
5. **Connect your input image** (browse to select your document photo)
6. **Run the workflow**
7. **Get your enhanced document!**

The processed images will be automatically saved to your storage folder.

## Example Use Cases

### Case 1: Business Card Scanning
You photograph a business card at an angle with your phone. The **Distortion Correction** block automatically straightens it, making all text clearly readable and properly aligned.

### Case 2: Book Page Digitization
You take a photo of an open book where the pages curve near the binding. The **Distortion Correction** block flattens the curve, creating a scan-like image perfect for reading or archiving.

### Case 3: Low-Light Document Capture
You photograph a document in poor lighting with uneven shadows. The **Illumination Correction** block removes shadows and balances the lighting, producing a clear, uniformly lit image.

### Case 4: Complete Document Enhancement
You capture a document that's both skewed and poorly lit. Chain both blocks together:
**Illumination Correction** → **Distortion Correction** → Perfect document!

### Case 5: Document Workflow Automation
Combine these blocks with other OOMOL blocks to create powerful automated workflows:
- Illumination Correction → Distortion Correction → Extract text (OCR) → Save to database
- Distortion Correction → Compress image → Email attachment
- Illumination Correction → Convert to PDF → Archive
- Combined corrections → Batch process folder → Generate searchable PDFs

## Technical Details (For Developers)

### Architecture

**Document Distortion Correction**:
- **Model**: GeoTr (Geometric Transformer) with U2-Net segmentation
- **Processing**: Automatic geometric transformation with perspective correction
- **Input Size**: 288x288 (with automatic resizing)

**Document Illumination Correction**:
- **Model**: IllTr (Illumination Transformer)
- **Processing**: Patch-based processing with 128x128 patches and 12.5% overlap
- **Method**: Transformer-based deep learning for lighting enhancement

**Common Infrastructure**:
- **Framework**: PyTorch
- **Inference**: CPU/GPU compatible
- **Model Loading**: Singleton pattern for efficient memory usage

### Dependencies
- PyTorch >= 2.8.0
- OpenCV (headless)
- NumPy < 2.0
- Pillow >= 11.3.0
- scikit-image >= 0.25.2
- timm >= 1.0.20

### Model Files
Pre-trained models are automatically loaded from `/oomol-driver/oomol-storage/model_pretrained/`:
- `seg.pth`: Document segmentation model (4.5MB)
- `geotr.pth`: Geometric transformation model (99MB)
- `illtr.pth`: Illumination correction model (50MB)

### Model Source
All models are sourced from the [DocTr project](https://huggingface.co/spaces/HaoFeng2019/DocTr) on HuggingFace.

## Support

For issues, questions, or feature requests, please contact the OOMOL community or create an issue in the project repository.

## License

This project is part of the OOMOL ecosystem. Please refer to the project license for usage terms.

## Credits

Developed by alwaysmavs | Powered by OOMOL Platform
