from typing import Dict, List, Optional
from torch import nn, Tensor

from .model.multimae import generate_smultimae_model as generate_smultimae_model_v1
from .configs.base_config import base_cfg


class RGBDModel(nn.Module):
    def __init__(self, cfg: base_cfg):
        super(RGBDModel, self).__init__()

        self.inputs = cfg.inputs
        self.outputs = cfg.outputs

        self.is_no_depth = cfg.is_inference_with_no_depth

        if cfg.model_version == 1:
            self.model, self.opt_params = generate_smultimae_model_v1(cfg)
        else:
            raise Exception(f"Unsupported model version {cfg.model_version}")

    def encode_decode(
        self,
        images: Tensor,
        depths: Optional[Tensor],
        gt_index_lst: Optional[List[int]] = None,
        max_gts_lst: Optional[List[int]] = None,
    ) -> Dict[str, Tensor]:
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input.

        Returns:
        {
            "sod": Tensor,
            "depth": Optional[Tensor],
            "rgb": Optional[tensor],
        }
        """
        inputs = {"rgb": images}
        if "depth" in self.inputs:
            inputs["depth"] = depths
        return self.model.forward(inputs, gt_index_lst, max_gts_lst)

    def forward(
        self,
        images: Tensor,
        depths: Optional[Tensor],
        gt_index_lst: Optional[List[int]] = None,
        max_gts_lst: Optional[List[int]] = None,
    ) -> Dict[str, Tensor]:
        return self.encode_decode(images, depths, gt_index_lst, max_gts_lst)

    def inference(
        self,
        images: Tensor,
        depths: Optional[Tensor],
        gt_index_lst: Optional[List[int]] = None,
        max_gts_lst: Optional[List[int]] = None,
    ) -> Dict[str, Tensor]:
        return self.encode_decode(images, depths, gt_index_lst, max_gts_lst)
