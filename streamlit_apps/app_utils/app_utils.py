import random
import time
from typing import Tuple, Union
import cv2
import numpy as np
from PIL import Image
from torch import nn, Tensor
import torch

num_format = "{:,}".format


def to_tensor(img: Image.Image) -> Tensor:
    # Convert the PIL image to a numpy array
    img_array = np.array(img)

    # Convert the numpy array to a PyTorch tensor
    img_tensor = torch.from_numpy(img_array)

    if len(img_tensor.shape) == 3:
        # Transpose the tensor to have the correct shape (C, H, W)
        img_tensor = img_tensor.permute(2, 0, 1)

    # Normalize the tensor values to be between 0 and 1
    img_tensor = img_tensor.float().div(255)

    return img_tensor


def count_parameters(model: nn.Module) -> str:
    """Count the number of parameters of a model"""
    return num_format(sum(p.numel() for p in model.parameters() if p.requires_grad))


class FrameRate:
    def __init__(self) -> None:
        self.c: int = 0
        self.start_time: float = None
        self.NO_FRAMES = 100
        self.fps: float = -1

    def reset(self) -> None:
        self.start_time = time.time()
        self.c = 0
        self.fps = -1

    def count(self) -> None:
        self.c += 1
        if self.c % self.NO_FRAMES == 0:
            self.c = 0
            end_time = time.time()
            self.fps = self.NO_FRAMES / (end_time - self.start_time)
            self.start_time = end_time

    def show_fps(self, image: np.ndarray) -> np.ndarray:
        if self.fps != -1:
            return cv2.putText(
                image,
                f"FPS {self.fps:.0f}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(255, 0, 0),
                thickness=2,
            )
        else:
            return image


class ImgContainer:
    img: np.ndarray = None  # raw image
    frame_rate: FrameRate = FrameRate()


def normalize(data: np.ndarray) -> np.ndarray:
    return (data - data.min()) / (data.max() - data.min() + 1e-8)


def get_size(image: Union[Image.Image, np.ndarray]) -> Tuple[int, int]:
    """Get resolution (w, h) of an image
    An input image can be Pillow Image or CV2 Image
    """
    if type(image) == np.ndarray:
        return (image.shape[1], image.shape[0])
    else:
        return image.size


def random_choice(p: float) -> bool:
    """Return True if random float <= p"""
    return random.random() <= p
