#region generated meta
import typing
class Inputs(typing.TypedDict):
    input_image: str
class Outputs(typing.TypedDict):
    output_image: typing.NotRequired[str]
#endregion

from oocana import Context
import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import sys
from PIL import Image
from functools import partial

# Handle imports for both package and standalone execution
try:
    from .model_code.IllTr import IllTr
except ImportError:
    # Add current directory to path for standalone execution
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    from model_code.IllTr import IllTr


def reload_model(model, path=""):
    """Load pretrained weights for IllTr model"""
    if not bool(path):
        return model
    else:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(path, map_location='cpu')
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        return model


# Global model instance (loaded once)
_model = None


def get_model():
    """Get or initialize the IllTr model (singleton pattern)"""
    global _model
    if _model is None:
        model_dir = '/oomol-driver/oomol-storage/model_pretrained'
        illtr_model_path = os.path.join(model_dir, 'illtr.pth')

        _model = IllTr()
        reload_model(_model, illtr_model_path)
        _model.eval()

    return _model


def pad_crop_img(img):
    """Divide image into overlapping patches for processing"""
    H = img.shape[0]
    W = img.shape[1]

    patchRes = 128
    pH = patchRes
    pW = patchRes
    ovlp = int(patchRes * 0.125)  # 32

    padH = (int((H - patchRes) / (patchRes - ovlp) + 1) * (patchRes - ovlp) + patchRes) - H
    padW = (int((W - patchRes) / (patchRes - ovlp) + 1) * (patchRes - ovlp) + patchRes) - W

    padImg = cv2.copyMakeBorder(img, 0, padH, 0, padW, cv2.BORDER_REPLICATE)

    ynum = int((padImg.shape[0] - pH) / (pH - ovlp)) + 1
    xnum = int((padImg.shape[1] - pW) / (pW - ovlp)) + 1

    totalPatch = np.zeros((ynum, xnum, patchRes, patchRes, 3), dtype=np.uint8)

    for j in range(0, ynum):
        for i in range(0, xnum):
            x = int(i * (pW - ovlp))
            y = int(j * (pH - ovlp))

            if j == (ynum-1) and i == (xnum-1):
                totalPatch[j, i] = img[-patchRes:, -patchRes:]
            elif j == (ynum-1):
                totalPatch[j, i] = img[-patchRes:, x:int(x + patchRes)]
            elif i == (xnum-1):
                totalPatch[j, i] = img[y:int(y + patchRes), -patchRes:]
            else:
                totalPatch[j, i] = padImg[y:int(y + patchRes), x:int(x + patchRes)]

    return totalPatch, padH, padW


def ill_correction(model, totalPatch):
    """Apply illumination correction to all patches"""
    totalPatch = totalPatch.astype(np.float32) / 255.0

    ynum = totalPatch.shape[0]
    xnum = totalPatch.shape[1]

    totalResults = np.zeros((ynum, xnum, 128, 128, 3), dtype=np.float32)

    with torch.no_grad():
        for j in range(0, ynum):
            for i in range(0, xnum):
                patchImg = torch.from_numpy(totalPatch[j, i]).permute(2, 0, 1)
                patchImg = patchImg.float().view(1, 3, 128, 128)

                output = model(patchImg)
                output = output.permute(0, 2, 3, 1).cpu().numpy()[0]

                output = output * 255.0
                output = output.astype(np.uint8)

                totalResults[j, i] = output

    return totalResults


def compose_patch(totalResults, padH, padW, img):
    """Reconstruct full image from corrected patches"""
    ynum = totalResults.shape[0]
    xnum = totalResults.shape[1]
    patchRes = totalResults.shape[2]

    ovlp = int(patchRes * 0.125)
    step = patchRes - ovlp

    resImg = np.zeros_like(img).astype('uint8')

    for j in range(0, ynum):
        for i in range(0, xnum):
            sy = int(j * step)
            sx = int(i * step)

            if j == 0 and i != (xnum-1):
                resImg[sy:(sy + patchRes), sx:(sx + patchRes)] = totalResults[j, i]
            elif i == 0 and j != (ynum-1):
                resImg[sy+10:(sy + patchRes), sx:(sx + patchRes)] = totalResults[j, i, 10:]
            elif j == (ynum-1) and i == (xnum-1):
                resImg[-patchRes+10:, -patchRes+10:] = totalResults[j, i, 10:, 10:]
            elif j == (ynum-1) and i == 0:
                resImg[-patchRes+10:, sx:(sx + patchRes)] = totalResults[j, i, 10:]
            elif j == (ynum-1) and i != 0:
                resImg[-patchRes+10:, sx+10:(sx + patchRes)] = totalResults[j, i, 10:, 10:]
            elif i == (xnum-1) and j == 0:
                resImg[sy:(sy + patchRes), -patchRes+10:] = totalResults[j, i, :, 10:]
            elif i == (xnum-1) and j != 0:
                resImg[sy+10:(sy + patchRes), -patchRes+10:] = totalResults[j, i, 10:, 10:]
            else:
                resImg[sy+10:(sy + patchRes), sx+10:(sx + patchRes)] = totalResults[j, i, 10:, 10:]

    resImg[0, :, :] = 255

    return resImg


def main(params: Inputs, context: Context) -> Outputs:
    """
    Main function: Correct illumination issues in document images

    Args:
        params: Input parameter dictionary with input_image path
        context: OOMOL context object

    Returns:
        Output result dictionary with corrected image path
    """
    input_image_path = params["input_image"]

    if not os.path.exists(input_image_path):
        raise ValueError(f"Input image not found: {input_image_path}")

    # Load image
    input_image = Image.open(input_image_path).convert('RGB')
    img = np.array(input_image)

    # Get model
    model = get_model()

    # Process image in patches
    totalPatch, padH, padW = pad_crop_img(img)
    totalResults = ill_correction(model, totalPatch)
    resImg = compose_patch(totalResults, padH, padW, img)

    # Generate output path
    input_basename = os.path.basename(input_image_path)
    input_name, input_ext = os.path.splitext(input_basename)
    output_filename = f"{input_name}_ill_corrected{input_ext}"

    # Save to oomol-storage
    output_dir = "/oomol-driver/oomol-storage"
    os.makedirs(output_dir, exist_ok=True)
    output_image_path = os.path.join(output_dir, output_filename)

    # Save corrected image
    output_image = Image.fromarray(resImg)
    output_image.save(output_image_path)

    return {"output_image": output_image_path}
