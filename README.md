# Document Image Correction

Transform your smartphone photos of documents into professional-quality scans with AI-powered correction technology.

## Overview

Have you ever taken a photo of a document with your phone, only to find it distorted, shadowed, or poorly lit? This project provides an intelligent solution that automatically enhances document images by fixing lighting issues and correcting geometric distortions—turning casual photos into clean, readable scans.

## What Does This Project Do?

This OOMOL workflow package uses advanced AI models to process document images through two main corrections:

1. **Illumination Correction**: Removes shadows, evens out lighting, and fixes exposure issues that occur when photographing documents under poor lighting conditions.

2. **Geometric Correction**: Straightens warped or curved documents, corrects perspective distortion, and flattens images taken from angles.

### Real-World Use Cases

- **Office Work**: Quickly digitize paper contracts, receipts, or forms using just your phone
- **Education**: Convert handwritten notes or textbook pages into clear digital copies
- **Research**: Archive historical documents or manuscripts with enhanced readability
- **Travel**: Scan important documents like passports or tickets without a scanner
- **Remote Work**: Share clear document images in emails or chat without needing scanning equipment

## Available Blocks

### Document Image Correction

**What it does**: This is the core processing block that takes a document photo and applies both illumination and geometric corrections to produce a professional-quality output.

**Inputs**:
- **Input Image**: Your original document photo (supports JPG, PNG, BMP, TIFF formats)
- **Enable Illumination Correction**: Option to turn on/off the lighting enhancement feature (recommended: enabled)

**Output**:
- **Enhanced Image**: A corrected, high-quality document image ready for use

**How to use**: Simply drag this block into your workflow, connect your input image, and the block will automatically process it. You can choose to enable or disable illumination correction based on your needs—if your document has good lighting but is warped or curved, you might only need geometric correction.

## How to Get Started

### For OOMOL Platform Users

1. **Import this package** into your OOMOL workspace
2. **Create a new flow** or use the provided test flow
3. **Add the Document Image Correction block** to your canvas
4. **Connect your input image** (from file upload or another block)
5. **Run the workflow** and view your enhanced document

### Example Workflow

The package includes a ready-to-use test workflow (`test-document-image-correction`) that demonstrates the complete process:

```
Input Image → Document Image Correction → Preview Enhanced Result
```

You can duplicate this workflow as a starting point for your own document processing pipelines.

## Technical Requirements

- **Platform**: OOMOL (version with Python 3.10-3.12 support)
- **Dependencies**: Automatically installed during setup
- **Processing Time**: Typically 2-10 seconds per image, depending on image size and hardware

## Key Features

✓ **Fully Automated**: No manual adjustments needed—the AI handles everything
✓ **High Quality**: Uses state-of-the-art deep learning models trained on thousands of document images
✓ **Flexible**: Can be used as a standalone tool or integrated into larger workflows
✓ **Batch Processing**: Process multiple documents by connecting this block in a loop
✓ **Format Support**: Works with all common image formats (JPG, PNG, BMP, TIFF)

## Understanding the Technology

This project uses transformer-based neural networks specifically designed for document enhancement:

- **Illumination Correction Model**: Analyzes lighting patterns and reconstructs evenly-lit versions of the document
- **Geometric Correction Model**: Detects document boundaries and applies intelligent warping to flatten and straighten the image

Both models have been pre-trained on large datasets and are optimized for speed and accuracy.

## Limitations and Best Practices

**For best results**:
- Ensure the entire document is visible in the photo
- Avoid extreme angles (greater than 45 degrees from vertical)
- Provide adequate ambient lighting (the correction works better with some base lighting)
- Use images with at least 800x600 resolution

**Current limitations**:
- Works best with single-page documents (not spreads or multiple pages)
- May struggle with heavily crumpled or torn documents
- Processing time increases with very high-resolution images (>5000px)

## Support and Contributions

This is an open-source project built for the OOMOL platform. If you encounter issues or have suggestions for improvements, please open an issue in the repository.

## License

This project is provided as-is for use with the OOMOL platform. Pre-trained AI models are included for non-commercial use.

---

**Version**: 0.0.1
**Author**: alwaysmavs
**Platform**: OOMOL
