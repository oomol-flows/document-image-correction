#region generated meta
import typing
class Inputs(typing.TypedDict):
    input_image: str
    enable_illumination_correction: bool
class Outputs(typing.TypedDict):
    output_image: typing.NotRequired[str]
#endregion

from oocana import Context
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
import sys
from PIL import Image

# Handle imports for both package and standalone execution
try:
    from .model_code.seg import U2NETP
    from .model_code.GeoTr import GeoTr
    from .model_code.IllTr import IllTr
except ImportError:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    from model_code.seg import U2NETP
    from model_code.GeoTr import GeoTr
    from model_code.IllTr import IllTr


class GeoTr_Seg(nn.Module):
    """Combined model for document segmentation and geometric correction"""
    def __init__(self):
        super(GeoTr_Seg, self).__init__()
        self.msk = U2NETP(3, 1)
        self.GeoTr = GeoTr(num_attn_layers=6)

    def forward(self, x):
        msk, _1, _2, _3, _4, _5, _6 = self.msk(x)
        msk = (msk > 0.5).float()
        x = msk * x

        bm = self.GeoTr(x)
        bm = (2 * (bm / 286.8) - 1) * 0.99

        return bm


def reload_geotr_model(model, path=""):
    """Load pretrained weights for GeoTr model"""
    if not bool(path):
        return model
    else:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(path, map_location='cpu')
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        return model


def reload_segmodel(model, path=""):
    """Load pretrained weights for segmentation model"""
    if not bool(path):
        return model
    else:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(path, map_location='cpu')
        pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if k[6:] in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        return model


def reload_illtr_model(model, path=""):
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


# Global model instances (loaded once)
_geo_model = None
_ill_model = None


def get_geo_model():
    """Get or initialize the geometric correction model (singleton pattern)"""
    global _geo_model
    if _geo_model is None:
        model_dir = '/oomol-driver/oomol-storage/model_pretrained'
        seg_model_path = os.path.join(model_dir, 'seg.pth')
        geotr_model_path = os.path.join(model_dir, 'geotr.pth')

        _geo_model = GeoTr_Seg()
        reload_segmodel(_geo_model.msk, seg_model_path)
        reload_geotr_model(_geo_model.GeoTr, geotr_model_path)
        _geo_model.eval()

    return _geo_model


def get_ill_model():
    """Get or initialize the illumination correction model (singleton pattern)"""
    global _ill_model
    if _ill_model is None:
        model_dir = '/oomol-driver/oomol-storage/model_pretrained'
        illtr_model_path = os.path.join(model_dir, 'illtr.pth')

        _ill_model = IllTr()
        reload_illtr_model(_ill_model, illtr_model_path)
        _ill_model.eval()

    return _ill_model


def pad_crop_img(img):
    """Divide image into overlapping patches for illumination correction"""
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


def apply_illumination_correction(img_array):
    """Apply illumination correction to the image"""
    ill_model = get_ill_model()

    # Process image in patches
    totalPatch, padH, padW = pad_crop_img(img_array)
    totalResults = ill_correction(ill_model, totalPatch)
    corrected_img = compose_patch(totalResults, padH, padW, img_array)

    return corrected_img


def apply_geometric_correction(img_array):
    """Apply geometric correction to the image"""
    geo_model = get_geo_model()

    # Normalize to [0, 1]
    im_ori = img_array.astype(np.float32) / 255.
    h, w, _ = im_ori.shape

    # Resize to model input size
    im = cv2.resize(im_ori, (288, 288))
    im = im.transpose(2, 0, 1)
    im = torch.from_numpy(im).float().unsqueeze(0)

    # Inference
    with torch.no_grad():
        bm = geo_model(im)
        bm = bm.cpu()
        bm0 = cv2.resize(bm[0, 0].numpy(), (w, h))
        bm1 = cv2.resize(bm[0, 1].numpy(), (w, h))
        bm0 = cv2.blur(bm0, (3, 3))
        bm1 = cv2.blur(bm1, (3, 3))
        lbl = torch.from_numpy(np.stack([bm0, bm1], axis=2)).unsqueeze(0)

        # Apply geometric correction using grid sampling
        out = F.grid_sample(
            torch.from_numpy(im_ori).permute(2, 0, 1).unsqueeze(0).float(),
            lbl,
            align_corners=True
        )
        img_geo = ((out[0] * 255).permute(1, 2, 0).numpy()).astype(np.uint8)

    return img_geo


def main(params: Inputs, context: Context) -> Outputs:
    """
    Main function: Complete document enhancement with optional illumination and geometric correction

    Args:
        params: Input parameter dictionary with input_image path and enable_illumination_correction flag
        context: OOMOL context object

    Returns:
        Output result dictionary with enhanced image path
    """
    input_image_path = params["input_image"]
    enable_ill_correction = params.get("enable_illumination_correction", True)

    if not os.path.exists(input_image_path):
        raise ValueError(f"Input image not found: {input_image_path}")

    # Load image
    input_image = Image.open(input_image_path).convert('RGB')
    img_array = np.array(input_image)

    # Step 1: Illumination correction (optional, but recommended first)
    if enable_ill_correction:
        img_array = apply_illumination_correction(img_array)

    # Step 2: Geometric correction (always applied)
    img_array = apply_geometric_correction(img_array)

    # Generate output path
    input_basename = os.path.basename(input_image_path)
    input_name, input_ext = os.path.splitext(input_basename)
    suffix = "_enhanced" if enable_ill_correction else "_geo_corrected"
    output_filename = f"{input_name}{suffix}{input_ext}"

    # Save to oomol-storage
    output_dir = "/oomol-driver/oomol-storage"
    os.makedirs(output_dir, exist_ok=True)
    output_image_path = os.path.join(output_dir, output_filename)

    # Save enhanced image
    output_image = Image.fromarray(img_array)
    output_image.save(output_image_path)

    return {"output_image": output_image_path}
