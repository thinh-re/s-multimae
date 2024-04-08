from functools import partial
import os
from typing import Dict, Optional, Tuple, List
import torch
from torch import nn
import math
import albumentations as A

from definition import PRETRAINED_BACKBONE
from .data_augmentation_config import DataAugmentationConfig


class base_cfg:
    def __init__(
        self,
        epoch: int,
        datasets_set: int,
        experiment_name: Optional[str] = None,
    ):
        self.experiment_name = experiment_name = (
            self.__class__.__name__ if experiment_name is None else experiment_name
        )
        self.datasets_set = datasets_set

        # Trainv3
        self.devices: List[int] = [0, 1]
        # How often to check the validation set. Pass a float in the range [0.0, 1.0] to check
        self.val_check_interval: float = 1.0

        # Perform a validation loop every after every N training epochs.
        self.check_val_every_n_epoch: int = 2

        self.precision = 16
        self.transform1 = [A.HorizontalFlip(p=0.5)]

        self.save_top_k = 2

        # ConvNeXtAdapter
        self.dec_kernel = 1  # decoder kernel size

        # Version 1: as usual
        # Version 2: mean, std
        self.model_version = 1

        self.visualized_num_dev_samples = 0

        # PytorchLightning Trainer
        self.sync_batchnorm = True

        self.normalized_depth: bool = True

        self.test_image_size: int = 224
        self.image_size: int = 224

        """Whether using fp16 instead of fp32 (default)"""
        self.is_fp16: bool = True

        self.is_padding: bool = (
            False  # deprecated due to randomly switch between padding and non-padding
        )

        # """For debug only"""
        # self.max_train_samples: Optional[int] = None
        # self.max_dev_samples: Optional[int] = None

        """Whether using padding for test"""
        self.is_padding_for_test: bool = False

        """Seed"""
        self.seed: int = 2022

        """ MultiMAE """
        self.decoder_depth: int = 4
        self.encoder_depth: int = 12
        self.is_inference_with_no_depth: bool = False
        self.inputs = ["rgb", "depth"]
        self.outputs = ["sod"]
        self.decoder_main_tasks: List[List[str]] = [["rgb"]]
        self.learnable_pos_emb: bool = False
        self.learnable_additional_gt_tokens: bool = False
        self.decoder_interpolate_mode: str = "bilinear"  # ['bilinear', 'nearest']
        self.dim_tokens: int = 768
        self.act_fn = partial(nn.ReLU, inplace=True)
        self.num_heads: int = 12
        self.freeze_encoder: bool = False

        """Data Augmentation"""
        self.data_augmentation_version: int = 2
        self.data_augmentation_config = DataAugmentationConfig()

        self.ckpt_path: Optional[str] = None
        self.description: str = ""  # Override this
        self.embed_dim: int = 6144

        """Pretrained Backbone"""
        self.pretrained_backbone: Optional[PRETRAINED_BACKBONE] = (
            PRETRAINED_BACKBONE.MULTIMAE
        )

        """
        Required only when self.pretrained_backbone in [PRETRAINED_BACKBONE.S_MULTIMAE, PRETRAINED_BACKBONE.LARGE_S_MULTIMAE].
        Example: 'v1.0.4_e499' stands for version 1.0.4, epoch 499, trained 500 epochs
        """
        self.pretrained_backbone_version: Optional[str] = None

        """Ground truth
        V1: 1 head, each head has 1 class, BCE
        V2: 1 head, each head has 5 classes, CE
        V3: 5 heads, each head has 1 class, BCE
        V4: 1 head, each head has 5 classes, BCE
        V5: additional global token indicates individual thinker
        """
        self.ground_truth_version = 1
        self.additional_gt_tokens_mlp_channels = []
        self.num_classes = 1
        self.actual_num_classes = 1

        self.is_cache = False

        """Learning rate
        LR strategy:
        V1: The ratio of unpretrained and pretrained is also 1:lr_scale
        V2: The ratio of unpretrained and pretrained is changed gradually from 1:lr_scale -> 1:1  
        """
        self.lr_strategy_version = 1
        self.lr: float
        self.end_lr: float = 1e-11
        self.lr_scale: int
        self.lr_power: float = 0.9

        # Deprecated from v3
        self.save_checkpoints_after_each_n_epochs: int = 10  # Not used in trainv3

        self.weight_decay = 0.05
        self.num_workers = 2
        self.num_epochs_every_restart = 100

        self.betas: Tuple[float, float] = (0.9, 0.999)

        self.input_patch_size: int = 16
        self.output_patch_size: int = 16  # must be a square of number

        """Warmup batchsize"""
        self.warmup_min_batch_size: Optional[int] = None
        self.warmup_epoch_batch_size: Optional[int] = None

        self.batch_size: int
        self.val_batch_size: int
        self.test_batch_size: int = 100
        self.nepochs: int

    def todict(self):
        d = dict()
        for k, v in self.__dict__.items():
            if not k.startswith("_"):
                d[k] = v
        return d

    @property
    def total_iters_per_epoch(self):
        return math.ceil(
            (self.num_training_samples_per_epoch)
            / (self.batch_size * len(self.devices))
        )
