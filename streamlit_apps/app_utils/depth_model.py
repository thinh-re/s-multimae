import os
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch import Tensor, nn

from .app_utils import count_parameters
from .device import device
from .dpt.models import DPTDepthModel


class BaseDepthModel:
    def __init__(self, image_size: int) -> None:
        self.image_size = image_size
        self.model: nn.Module = None

    def forward(self, image: Tensor) -> Tensor:
        """Perform forward inference for an image
        Input image of shape [c, h, w]
        Return of shape [c, h, w]
        """
        raise NotImplementedError()

    def batch_forward(self, images: Tensor) -> Tensor:
        """Perform forward inference for a batch of images
        Input images of shape [b, c, h, w]
        Return of shape [b, c, h, w]"""
        raise NotImplementedError()

    def get_number_of_parameters(self) -> int:
        return count_parameters(self.model)


class DPTDepth(BaseDepthModel):
    def __init__(self, image_size: int) -> None:
        super().__init__(image_size)
        print("DPTDepthconstructor")
        weights_fname = "omnidata_rgb2depth_dpt_hybrid.pth"
        weights_path = os.path.join("weights", weights_fname)
        if not os.path.isfile(weights_path):
            from huggingface_hub import hf_hub_download
            hf_hub_download(repo_id="RGBD-SOD/S-MultiMAE", filename=weights_fname)
            os.system(f"mv {weights_fname} weights")
        omnidata_ckpt = torch.load(
            weights_path,
            map_location="cpu",
        )
        
        self.model = DPTDepthModel()
        self.model.load_state_dict(omnidata_ckpt)
        self.model: DPTDepthModel = self.model.to(device).eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.image_size, self.image_size),
                    interpolation=TF.InterpolationMode.BICUBIC,
                ),
                transforms.Normalize(
                    (0.5, 0.5, 0.5),
                    (0.5, 0.5, 0.5),
                ),
            ]
        )

    def forward(self, image: Tensor) -> Tensor:
        depth_model_input = self.transform(image.unsqueeze(0))
        return self.model.forward(depth_model_input.to(device)).squeeze(0)

    def batch_forward(self, images: Tensor) -> Tensor:
        images: Tensor = TF.resize(
            images,
            (self.image_size, self.image_size),
            interpolation=TF.InterpolationMode.BICUBIC,
        )
        depth_model_input = (images - 0.5) / 0.5
        return self.model(depth_model_input.to(device))
