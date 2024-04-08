from collections import defaultdict
import os
from typing import Any, Dict, List, Optional, Tuple
import cv2
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np

from .configs.base_config import base_cfg
from .rgbd_model import RGBDModel


class ModelPL(pl.LightningModule):
    def __init__(self, cfg: base_cfg):
        super().__init__()
        self.cfg = cfg
        self.model = RGBDModel(cfg)

    def forward(self, images: Tensor, depths: Tensor):
        return self.model.forward(images, depths)

    def __inference_v1(
        self, outputs: Dict[str, Tensor], image_sizes: List[Tuple[int, int]]
    ):
        res_lst: List[List[np.ndarray]] = []
        for output, image_size in zip(outputs["sod"], image_sizes):
            output: Tensor = F.interpolate(
                output.unsqueeze(0),
                size=(image_size[1], image_size[0]),
                mode="bilinear",
                align_corners=False,
            )
            res: np.ndarray = output.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            if self.cfg.is_fp16:
                res = np.float32(res)
            res_lst.append([(res * 255).astype(np.uint8)])
        return res_lst

    def __inference_v2(
        self, outputs: Dict[str, Tensor], image_sizes: List[Tuple[int, int]]
    ):
        res_lst: List[List[np.ndarray]] = []
        for output, image_size in zip(outputs["sod"], image_sizes):
            output: Tensor = F.interpolate(
                output.unsqueeze(0),
                size=(image_size[1], image_size[0]),
                mode="bilinear",
                align_corners=False,
            )
            res: np.ndarray = torch.argmax(output, dim=1).cpu().numpy().squeeze()
            res_lst.append([res])
        return res_lst

    def __inference_v3v5(
        self, outputs: Dict[str, Tensor], image_sizes: List[Tuple[int, int]]
    ):
        res_lst: List[List[np.ndarray]] = []
        for bi, image_size in enumerate(image_sizes):
            res_lst_per_sample: List[np.ndarray] = []
            for i in range(self.cfg.num_classes):
                pred = outputs[f"sod{i}"][bi]
                pred: Tensor = F.interpolate(
                    pred.unsqueeze(0),
                    size=(image_size[1], image_size[0]),
                    mode="bilinear",
                    align_corners=False,
                )
                res: np.ndarray = pred.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                if self.cfg.is_fp16:
                    res = np.float32(res)
                res_lst_per_sample.append((res * 255).astype(np.uint8))
            res_lst.append(res_lst_per_sample)
        return res_lst

    @torch.no_grad()
    def inference(
        self,
        image_sizes: List[Tuple[int, int]],
        images: Tensor,
        depths: Optional[Tensor],
        max_gts: Optional[List[int]],
    ) -> List[List[np.ndarray]]:
        self.model.eval()
        assert len(image_sizes) == len(
            images
        ), "The number of image_sizes must equal to the number of images"
        gpu_images: Tensor = images.to(self.device)
        gpu_depths: Tensor = depths.to(self.device)

        if self.cfg.ground_truth_version == 6:
            with torch.cuda.amp.autocast(enabled=self.cfg.is_fp16):
                outputs: Dict[str, Tensor] = dict()
                for i in range(self.cfg.num_classes):
                    outputs[f"sod{i}"] = self.model.inference(
                        gpu_images, gpu_depths, [i] * gpu_images.shape[0], max_gts
                    )["sod"]
            return self.__inference_v3v5(outputs, image_sizes)
        else:
            raise Exception(
                f"Unsupported ground_truth_version {self.cfg.ground_truth_version}"
            )
