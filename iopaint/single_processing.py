import base64
import io
import json
from pathlib import Path
from typing import Dict, Optional

import cv2
import psutil
from PIL import Image
from loguru import logger
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)

from iopaint.helper import pil_to_bytes_single
from iopaint.model.utils import torch_gc
from iopaint.model_manager import ModelManager
from iopaint.schema import InpaintRequest
import numpy as np


def glob_images(path: Path) -> Dict[str, Path]:
    # png/jpg/jpeg
    if path.is_file():
        return {path.stem: path}
    elif path.is_dir():
        res = {}
        for it in path.glob("*.*"):
            if it.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                res[it.stem] = it
        return res


# def batch_inpaint(
#     model: str,
#     device,
#     image: Path,
#     mask: Path,
#     config: Optional[Path] = None,
#     concat: bool = False,
# ):
#     if config is None:
#         inpaint_request = InpaintRequest()
#     else:
#         with open(config, "r", encoding="utf-8") as f:
#             inpaint_request = InpaintRequest(**json.load(f))
#
#     model_manager = ModelManager(name=model, device=device)
#
#     img = cv2.imread(str(image))
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
#     mask_img = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
#
#     if mask_img.shape[:2] != img.shape[:2]:
#         mask_img = cv2.resize(
#             mask_img,
#             (img.shape[1], img.shape[0]),
#             interpolation=cv2.INTER_NEAREST,
#         )
#
#     mask_img[mask_img >= 127] = 255
#     mask_img[mask_img < 127] = 0
#
#     # bgr
#     inpaint_result = model_manager(img, mask_img, inpaint_request)
#     inpaint_result = cv2.cvtColor(inpaint_result, cv2.COLOR_BGR2RGB)
#
#     if concat:
#         mask_img = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2RGB)
#         inpaint_result = cv2.hconcat([img, mask_img, inpaint_result])
#
#         # Convert the NumPy array to PIL Image
#     pil_image = Image.fromarray(inpaint_result)
#
#     # Encode the PIL Image as base64 string
#     with io.BytesIO() as output_buffer:
#         pil_image.save(output_buffer, format='PNG')
#         base64_image = base64.b64encode(output_buffer.getvalue()).decode('utf-8')
#
#     return base64_image

def batch_inpaint(
    model: str,
    device,
    input_base64: str,
    mask_base64: str,
    config_base64: Optional[str] = None,
    concat: bool = False,
):
    if config_base64 is None:
        inpaint_request = InpaintRequest()
    else:
        config_json = base64.b64decode(config_base64)
        inpaint_request = InpaintRequest(**json.loads(config_json))

    model_manager = ModelManager(name=model, device=device)

    # Decode input image from base64
    input_image_data = base64.b64decode(input_base64)
    input_image = cv2.imdecode(np.frombuffer(input_image_data, np.uint8), cv2.IMREAD_COLOR)

    # Decode mask image from base64
    mask_image_data = base64.b64decode(mask_base64)
    mask_image = cv2.imdecode(np.frombuffer(mask_image_data, np.uint8), cv2.IMREAD_GRAYSCALE)

    if mask_image.shape[:2] != input_image.shape[:2]:
        mask_image = cv2.resize(
            mask_image,
            (input_image.shape[1], input_image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    mask_image[mask_image >= 127] = 255
    mask_image[mask_image < 127] = 0

    # Run inpainting
    inpaint_result = model_manager(input_image, mask_image, inpaint_request)

    if concat:
        mask_image = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2RGB)
        inpaint_result = cv2.hconcat([input_image, mask_image, inpaint_result])

    # Convert NumPy array to PIL Image
    pil_image = Image.fromarray(inpaint_result)

    # Encode PIL Image to base64 string
    with io.BytesIO() as output_buffer:
        pil_image.save(output_buffer, format='PNG')
        base64_image = base64.b64encode(output_buffer.getvalue()).decode('utf-8')

    return base64_image