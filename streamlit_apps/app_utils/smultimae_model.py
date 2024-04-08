import numpy as np
from torch import Tensor
from torchvision.transforms import Resize

from s_multimae.model_pl import ModelPL
from s_multimae.configs.base_config import base_cfg

from .base_model import BaseRGBDModel


class RGBDSMultiMAEModel(BaseRGBDModel):
    def __init__(self, cfg: base_cfg, model: ModelPL):
        """Wrapper of RGBDModel"""
        super(RGBDSMultiMAEModel, self).__init__()
        self.model: ModelPL = model
        self.cfg = cfg
        self.resize = Resize([self.cfg.image_size, self.cfg.image_size])

    def inference(
        self,
        image: Tensor,
        depth: Tensor,
        origin_shape: np.array,
        num_sets_of_salient_objects: int = 1,
    ) -> np.ndarray:
        # 1. Preprocessing
        images = image.unsqueeze(0)
        depths = depth.unsqueeze(0)

        # images = self.resize(images)
        # depths = self.resize(depths)

        # 2. Inference
        images, depths = images.to(self.model.device), depths.to(self.model.device)
        if self.cfg.ground_truth_version == 6:
            self.cfg.num_classes = num_sets_of_salient_objects
        res = self.model.inference(
            [[origin_shape[2], origin_shape[1]]],
            images,
            depths,
            [num_sets_of_salient_objects],
        )
        return res[0]
