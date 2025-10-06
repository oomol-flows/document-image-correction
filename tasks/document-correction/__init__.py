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
except ImportError:
    # Add current directory to path for standalone execution
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    from model_code.seg import U2NETP
    from model_code.GeoTr import GeoTr


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


def reload_model(model, path=""):
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


# Global model instance (loaded once)
_model = None


def get_model():
    """Get or initialize the model (singleton pattern)"""
    global _model
    if _model is None:
        model_dir = '/oomol-driver/oomol-storage/model_pretrained'
        seg_model_path = os.path.join(model_dir, 'seg.pth')
        geotr_model_path = os.path.join(model_dir, 'geotr.pth')

        _model = GeoTr_Seg()
        reload_segmodel(_model.msk, seg_model_path)
        reload_model(_model.GeoTr, geotr_model_path)
        _model.eval()

    return _model


def main(params: Inputs, context: Context) -> Outputs:
    """
    Main function: Correct distorted document images

    Args:
        params: Input parameter dictionary with input_image path
        context: OOMOL context object

    Returns:
        Output result dictionary with corrected image path
    """
    input_image_path = params["input_image"]

    if not os.path.exists(input_image_path):
        raise ValueError(f"Input image not found: {input_image_path}")

    # Load and preprocess image
    input_image = Image.open(input_image_path).convert('RGB')
    im_ori = np.array(input_image)[:, :, :3] / 255.
    h, w, _ = im_ori.shape

    # Resize to model input size
    im = cv2.resize(im_ori, (288, 288))
    im = im.transpose(2, 0, 1)
    im = torch.from_numpy(im).float().unsqueeze(0)

    # Get model
    model = get_model()

    # Inference
    with torch.no_grad():
        bm = model(im)
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

    # Generate output path
    input_basename = os.path.basename(input_image_path)
    input_name, input_ext = os.path.splitext(input_basename)
    output_filename = f"{input_name}_corrected{input_ext}"

    # Save to oomol-storage
    output_dir = "/oomol-driver/oomol-storage"
    os.makedirs(output_dir, exist_ok=True)
    output_image_path = os.path.join(output_dir, output_filename)

    # Save corrected image
    output_image = Image.fromarray(img_geo)
    output_image.save(output_image_path)

    return {"output_image": output_image_path}
