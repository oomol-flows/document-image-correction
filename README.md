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
✓ **State-of-the-Art Performance**: Based on ACM MM 2021 research, achieving 15% absolute improvement over previous methods with 20.02% Character Error Rate
✓ **Dual Correction**: Combines geometric unwarping and illumination correction in one pipeline
✓ **Global Context Understanding**: Uses transformer self-attention mechanism to capture document-wide patterns
✓ **Flexible**: Can be used as a standalone tool or integrated into larger workflows
✓ **Batch Processing**: Process multiple documents by connecting this block in a loop
✓ **Format Support**: Works with all common image formats (JPG, PNG, BMP, TIFF)

## Understanding the Technology

This project is powered by **DocTr (Document Image Transformer)**, a state-of-the-art deep learning framework published at ACM MM 2021 as an oral paper. The technology uses transformer-based neural networks specifically designed for document enhancement:

- **GeoTr (Geometric Unwarping Transformer)**: Corrects geometric distortions by detecting document boundaries and applying intelligent warping to flatten and straighten warped or curved documents. Trained on Doc3D and DTD datasets.

- **IllTr (Illumination Correction Transformer)**: Analyzes lighting patterns and reconstructs evenly-lit versions of documents, removing shadows and fixing exposure issues. Trained on the DocProj dataset.

Both models leverage transformer architectures for superior performance and have been pre-trained on large-scale datasets, optimized for both speed and accuracy.

### Research Background

DocTr addresses two critical challenges in document image processing:
1. **Geometric distortion** caused by document warping and camera perspective
2. **Illumination variation** from uneven lighting conditions during capture

The framework achieves state-of-the-art performance on the DocUNet Benchmark, evaluated using metrics including MS-SSIM, Local Distortion (LD), and OCR accuracy.

**Research Paper**: [DocTr: Document Image Transformer for Geometric Unwarping and Illumination Correction](https://arxiv.org/abs/2110.12942) (ACM MM 2021, Oral)

**Original Implementation**: [fh2019ustc/DocTr](https://github.com/fh2019ustc/DocTr)

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

## Citation

If you use this work in your research or applications, please cite the original DocTr paper:

```bibtex
@inproceedings{feng2021doctr,
  title={DocTr: Document Image Transformer for Geometric Unwarping and Illumination Correction},
  author={Feng, Hao and Wang, Yuechen and Zhou, Wengang and Deng, Jiajun and Li, Houqiang},
  booktitle={Proceedings of the 29th ACM International Conference on Multimedia},
  pages={273--281},
  year={2021}
}
```

## Acknowledgments

This OOMOL implementation is based on the [DocTr](https://github.com/fh2019ustc/DocTr)

**Contact for Commercial Use**: For commercial applications of the original DocTr technology, please contact Professor Wengang Zhou or Hao Feng (see [original repository](https://github.com/fh2019ustc/DocTr)).

## Support and Contributions

This is an open-source project built for the OOMOL platform. If you encounter issues or have suggestions for improvements, please open an issue in the repository.

## License

This project is provided as-is for use with the OOMOL platform. Pre-trained AI models are included for non-commercial use. For commercial usage, please refer to the original DocTr project.

---

**Version**: 0.0.1
**Author**: alwaysmavs
**Platform**: OOMOL
