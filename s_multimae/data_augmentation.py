from torch import nn

from .configs.base_config import base_cfg
from .da.dav6 import DataAugmentationV6
from .da.base_da import BaseDataAugmentation


def get_data_augmentation(
    cfg: base_cfg,
    image_size: int,
    is_padding: bool,
) -> BaseDataAugmentation:
    if cfg.data_augmentation_version == 6:
        print("Using DataAugmentationV6")
        return DataAugmentationV6(cfg)
    else:
        raise NotImplementedError(
            f"Unsupported DataAugmentation version {cfg.data_augmentation_version}"
        )
