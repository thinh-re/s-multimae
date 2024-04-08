from typing import List, Optional, Tuple
from PIL import Image
import numpy as np
from torchvision import transforms
import albumentations as A
import torch
from torch import Tensor

from ..configs.base_config import base_cfg
from .base_da import BaseDataAugmentation


class DataAugmentationV6(BaseDataAugmentation):
    def __init__(
        self,
        cfg: base_cfg,
        is_padding=True,
    ):
        super().__init__()
        self.image_size = cfg.image_size
        self.is_padding = is_padding
        self.cfg = cfg

        self.to_tensor = transforms.ToTensor()

        self.additional_targets = {
            "depth": "image",
            "gt": "mask",
            "ranking_gt": "mask",
            "multi_gts": "mask",
        }

        # For rgb+depth+gt
        self.transform1 = A.Compose(
            cfg.transform1,
            additional_targets=self.additional_targets,
        )

        # For rgb only
        self.transform2 = A.Compose(
            [
                A.GaussianBlur(p=0.5, blur_limit=(3, 19)),
                A.RandomBrightnessContrast(p=0.5),
                A.ColorJitter(p=0.5),
            ]
        )

        # For depth only
        self.transform3 = A.Compose([A.GaussianBlur(p=0.5, blur_limit=(3, 37))])

        # For rgb+depth+gt
        self.transform4 = A.Compose(
            [A.Resize(self.image_size, self.image_size)],
            additional_targets=self.additional_targets,
            is_check_shapes=False,
        )

        # For rgb only
        self.transform5 = A.Compose([A.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

        # For depth only
        self.transform6 = A.Compose([A.Normalize(0.5, 0.5)])

    def forward(
        self,
        image: Image.Image,
        depth: Image.Image,
        gt: Optional[Image.Image] = None,
        ranking_gt: Optional[Image.Image] = None,
        multi_gts: Optional[List[Image.Image]] = None,
        is_transform: bool = True,  # is augmented?
        is_debug: bool = False,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        ## 1. Convert to numpy array: image, depth, gt, ranking_gts
        image = np.array(image)
        depth = np.array(depth)
        d = dict(image=image, depth=depth)
        if gt is not None:
            gt = np.array(gt)
            d["gt"] = gt

        if not is_transform:
            # Dev or Test
            d = self.transform4(**d)
            d["image"] = self.transform5(image=d["image"])["image"]
            # d["depth"] = self.transform6(image=depth)["image"]
            if gt is not None:
                return self.to_tensors([d["image"], d["depth"], d["gt"]])
            else:
                return self.to_tensors([d["image"], d["depth"]])

        d["depth"] = 255 - d["depth"]  # inverse depth

        # if ranking_gt is not None and multi_gts is not None:
        #     print('[WARN] Both ranking_gt and multi_gts are not none, but we prioritize multi_gts')

        if ranking_gt is not None:
            ranking_gt = np.array(ranking_gt)

        if multi_gts is not None:
            multi_gts = np.stack(multi_gts, axis=2)
            d["multi_gts"] = multi_gts

        ## 2. First transformation for image (Contrast, GaussianBlur,...), depth (GaussianBlur,...)
        d["image"] = self.transform2(image=d["image"])["image"]
        d["depth"] = self.transform3(image=d["depth"])["image"]

        ## 3. Transformation defined in config: change perspective, rotation, size, ...
        d = self.transform1(**d)

        ## 4. Resize
        d = self.transform4(**d)

        ## Just backup image before normalizing it
        if is_debug:
            unnormalized_image = d["image"]

        ## 6. Construct response
        d["depth"] = 255 - d["depth"]  # inverse depth
        d["image"] = self.transform5(image=d["image"])["image"]
        # d["depth"] = self.transform6(image=depth)["image"]
        rs = self.to_tensors([d["image"], d["depth"]])
        if multi_gts is not None:
            rs += self.to_tensors([d["multi_gts"]])
        elif ranking_gt is not None:
            rs += [torch.from_numpy(d["ranking_gt"]).to(torch.long)]
        else:
            rs += self.to_tensors([d["gt"]])

        ## 7. For debug only
        if is_debug:
            rs.append(unnormalized_image)

            if ranking_gt is not None:
                ranking_gts = []
                for i in range(self.cfg.num_classes):
                    ranking_gts.append(
                        np.array(d["ranking_gt"] == i).astype(np.uint8) * 255
                    )
                rs.append(ranking_gts)
            if multi_gts is not None:
                rs.append(d["multi_gts"])

        return rs

    def to_tensors(self, lst: List[Tensor]) -> List[Tensor]:
        return [self.to_tensor(e) for e in lst]
