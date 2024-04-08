from PIL import Image
from glob import glob
import random
from typing import Dict, List
from torch import nn, Tensor
import os, shutil
import torch
import matplotlib.pyplot as plt
import numpy as np
import gc, cv2

from .visualizer import post_processing_depth

"""
This module should not depend on other s_multimae modules.
"""

num_format = "{:,}".format


def list_dirs(dir_root: str) -> List[str]:
    return list(
        sorted(
            [
                item
                for item in os.listdir(dir_root)
                if os.path.isdir(f"{dir_root}/{item}")
            ]
        )
    )


def clean_cache() -> None:
    torch.cuda.empty_cache()
    gc.collect()


def count_parameters(model: nn.Module) -> str:
    """Count the number of learnable parameters of a model"""
    return num_format(sum(p.numel() for p in model.parameters() if p.requires_grad))


def ranking_gts_to_dict(
    ranking_gts: List[np.ndarray | str],
) -> Dict[str, np.ndarray | str]:
    """
    Return:
    dict(
        gt0=ranking_gts[0],
        gt1=ranking_gts[1],
        gt2=ranking_gts[2],
        gt3=ranking_gts[3],
        gt4=ranking_gts[4],
    )
    """
    return {f"gt{i}": v for i, v in enumerate(ranking_gts)}


def dict_to_ranking_gts(d: Dict[str, np.ndarray], l=5) -> List[np.ndarray]:
    """
    Return: [ranking_gts["gt0"], ranking_gts["gt1"], ...]
    """
    return [d[f"gt{i}"] for i in range(l)]


def random_choice(p: float) -> bool:
    """Return True if random float <= p"""
    return random.random() <= p


def fname_without_ext(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0]


def list_files(
    dirpaths: List[str] = [
        "datasets/v1/train/RGB",
        "datasets/v1/train/GT",
        "datasets/v1/train/depths",
    ],
) -> List[List[str]]:
    assert len(dirpaths) >= 1, "dirnames must contain at least 1 item"

    fullpaths_lst: List[List[str]] = []
    names_lst: List[List[str]] = []

    for dirname in dirpaths:
        fullpaths = list(sorted(glob(os.path.join(dirname, "*"))))
        names = [fname_without_ext(fullpath) for fullpath in fullpaths]
        fullpaths_lst.append(fullpaths)
        names_lst.append(names)

    rs: List[List[str]] = [fullpaths_lst[0]] + [[] for _ in range(len(dirpaths) - 1)]

    # Ensure integrity
    assert (
        len(set([len(e) for e in names_lst])) == 1
    ), f"Data is not integrity {[len(e) for e in names_lst]} | dirpath = {dirpaths}"

    for name in names_lst[0]:
        for i, names in enumerate(names_lst[1:]):
            idx = names.index(name)
            rs[i + 1].append(fullpaths_lst[i + 1][idx])

    return rs


def scale_saliency_maps(inputs: Tensor) -> Tensor:
    """Input: Tensor, shape of (B, C, H, W)"""
    min_v = (
        torch.min(torch.flatten(inputs, 1), dim=1)[0]
        .unsqueeze(1)
        .unsqueeze(1)
        .unsqueeze(1)
    )
    max_v = (
        torch.max(torch.flatten(inputs, 1), dim=1)[0]
        .unsqueeze(1)
        .unsqueeze(1)
        .unsqueeze(1)
    )
    return (inputs - min_v) / (max_v - min_v + 1e-8)


def get_epoch_from_ckpt_path(ckpt_path: str) -> int:
    """Example ckpt_path
    os.path.join(experiment_dir_path, 'cfgv2.3', 'checkpoint_100.pt')
    """
    return int(ckpt_path.split("_")[-1].split(".")[0])


def clean_dir(dir_path: str) -> None:
    """Remove a directory if existed and create an empty directory"""
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)


def get_sota_type(experiment_name: str) -> int:
    """0 for SOTAs, 4 for experiment version 4, e.g. ..."""
    if "cfgv" not in experiment_name:
        return 0

    half_right = experiment_name.split("cfgv")[1]
    return int(half_right.split("_")[0])


def hex_to_rgb(hex: str) -> np.ndarray:
    """Convert hex color to rgb color

    Args:
        hex (str): "#00f900"

    Returns:
        np.ndarray: numpy array of rgb color
    """
    hex = hex[1:]
    rgb = []
    for i in (0, 2, 4):
        decimal = int(hex[i : i + 2], 16)
        rgb.append(decimal)

    return (np.array(rgb) / 255.0)[::-1]


def normalize(data: np.ndarray) -> np.ndarray:
    return (data - data.min()) / (data.max() - data.min() + 1e-8)


def post_processing_depth(depth_path: str) -> np.ndarray:
    depth = np.array(Image.open(depth_path).convert("L"))
    depth = (normalize(depth) * 255).astype(np.uint8)
    return cv2.applyColorMap(depth, cv2.COLORMAP_SUMMER)


def convert_batch_tensors_to_numpy_images(images: Tensor) -> np.ndarray:
    """images of shape (batch_size, channels, width, height)"""
    images = torch.permute(images, (0, 2, 3, 1))
    images = images.numpy()
    if images.shape[3] == 1:
        return np.squeeze(images, axis=3)
    else:
        return images


def join_horizontally(lst: List[np.ndarray]) -> np.ndarray:
    return np.concatenate(lst, axis=1)


def join_vertically(lst: List[np.ndarray]) -> np.ndarray:
    return np.concatenate(lst, axis=0)


def plot_batch_of_pairs(
    images: Tensor,
    depths: Tensor,
    gts: Tensor,
    save_file_path: str,
) -> None:
    images = convert_batch_tensors_to_numpy_images(images)
    depths = convert_batch_tensors_to_numpy_images(depths)
    gts = convert_batch_tensors_to_numpy_images(gts)
    batch_size = images.shape[0]
    samples: List[np.ndarray] = []

    # fig, axes = plt.subplots(batch_size, 3, figsize=(3*batch_size, 20)) # (number of images, 3)
    for i in range(batch_size):
        samples.append(
            join_horizontally(
                [
                    ((images[i] + 1.0) / 2 * 255).astype(np.uint8),
                    post_processing_depth(depths[i]),
                    post_processing_depth(gts[i]),
                ]
            )
        )
        # axes[i, 0].imshow(images[i])
        # axes[i, 1].imshow(depths[i])
        # axes[i, 2].imshow(gts[i])
    # plt.show()

    final = join_vertically(samples)
    cv2.imwrite(save_file_path, cv2.cvtColor(final, cv2.COLOR_RGB2BGR))
    print(f"Saved to file {save_file_path}")


def plot_pairs(image: np.ndarray, depth: np.ndarray, gt: np.ndarray) -> None:
    batch_size = 1
    fig, axes = plt.subplots(
        batch_size, 3, figsize=(3 * batch_size, 20)
    )  # (number of images, 3)
    for i in range(batch_size):
        axes[i, 0].imshow(image)
        axes[i, 1].imshow(depth)
        axes[i, 2].imshow(gt)
    plt.show()
