from typing import List, Optional, Tuple, Union
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch import Tensor, nn
import torch
from skimage.filters import threshold_otsu

from s_multimae.da.base_da import BaseDataAugmentation
from s_multimae.model_pl import ModelPL
from s_multimae.visualizer import apply_vis_to_image

from .base_model import BaseRGBDModel
from .app_utils import get_size, normalize
from .depth_model import BaseDepthModel


# Environment
torch.set_grad_enabled(False)
from .device import device

print(f"device: {device}")


def post_processing_depth(depth: np.ndarray) -> np.ndarray:
    depth = (normalize(depth) * 255).astype(np.uint8)
    return cv2.applyColorMap(depth, cv2.COLORMAP_OCEAN)


def base_inference(
    depth_model: BaseDepthModel,
    sod_model: BaseRGBDModel,
    da: BaseDataAugmentation,
    raw_image: Union[Image.Image, np.ndarray],
    raw_depth: Optional[Union[Image.Image, np.ndarray]] = None,
    color: np.ndarray = None,
    num_sets_of_salient_objects: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inference a pair of rgb image and depth image
    if depth image is not provided, the depth_model will predict a depth image based on image
    """
    origin_size = get_size(raw_image)

    # Predict depth
    image = TF.to_tensor(raw_image)
    origin_shape = image.shape
    if raw_depth is None:
        depth: Tensor = depth_model.forward(image)
    else:
        depth = TF.to_tensor(raw_depth)

    # Preprocessing
    image, depth = da.forward(
        raw_image, depth.cpu().detach().squeeze(0).numpy(), is_transform=False
    )

    # Inference
    sms = sod_model.inference(image, depth, origin_shape, num_sets_of_salient_objects)

    # Postprocessing
    sods = []

    binary_masks = []

    for sm in sms:
        binary_mask = np.array(sm)
        t = threshold_otsu(binary_mask)
        binary_mask[binary_mask < t] = 0.0
        binary_mask[binary_mask >= t] = 1.0

        sod = apply_vis_to_image(np.array(raw_image), binary_mask, color)
        sods.append(sod)

        binary_mask = np.array(binary_mask * 255, dtype=np.uint8)
        binary_mask = np.stack((binary_mask, binary_mask, binary_mask), axis=-1)
        binary_masks.append(binary_mask)

    depth = depth.permute(1, 2, 0).detach().cpu().numpy()
    depth = cv2.resize(depth, origin_size)
    depth = post_processing_depth(depth)

    return depth, sods, [e / 255.0 for e in sms], binary_masks


def transform_images(inputs: List[Image.Image], transform: nn.Module) -> Tensor:
    if len(inputs) == 1:
        return transform(inputs[0]).unsqueeze(0)
    return torch.cat([transform(input).unsqueeze(0) for input in inputs])
