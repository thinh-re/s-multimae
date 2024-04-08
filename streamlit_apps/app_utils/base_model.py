from typing import List
import numpy as np
from torch import Tensor, nn


class BaseRGBDModel(nn.Module):
    def __init__(self):
        super(BaseRGBDModel, self).__init__()
        """
        Requirements:
        1. Construct a model
        2. Load pretrained weights
        3. Load model into device
        4. Construct preprocessing
        """

    def inference(
        self,
        image: Tensor,
        depth: Tensor,
        origin_shape: np.array,
    ) -> List[np.ndarray]:
        """
        Given:
        - An image (Tensor) with original shape [c, h, w]
        - A depth image (Tensor) with a shape of [c, h, w], do not need to be the same shape as image

        Requirements:
        1. Preprocessing
        2. Inference
        3. Return saliency maps np.float32 between 0.0 and 1.0,
           with the same size as original size

        """
        raise NotImplementedError()

    def batch_inference(
        self,
        images: Tensor,
        depths: Tensor,
    ) -> List[np.ndarray]:
        """
        Given:
        - A batch of images (Tensor) with original shape [b, c, h, w]
        - A batch of depths (Tensor) with a shape of [b, c, h, w], do not need to be the same shape as image

        Requirements:
        1. Preprocessing
        2. Inference
        3. Return saliency maps np.float32 between 0.0 and 1.0,
           with the same size as original size

        """
        raise NotImplementedError()
