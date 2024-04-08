import albumentations as A
import cv2

from typing import Optional
from definition import PRETRAINED_BACKBONE
from ..base_config import base_cfg


class cfgv4_0_2006(base_cfg):
    def __init__(self, epoch: Optional[int] = None):
        super().__init__(epoch, datasets_set=1)

        self.check_val_every_n_epoch = 1
        self.num_workers = 4
        self.devices = [0]

        self.description = "ViT Large [DoT]"

        """MultiMAE"""
        self.pretrained_backbone = PRETRAINED_BACKBONE.LARGE_S_MULTIMAE
        self.pretrained_backbone_version = "v2.0.5-pr"

        # Large MAE
        self.dim_tokens = 1024
        self.encoder_depth = 24
        self.num_heads = 16

        self.clip_grad = None
        self.normalized_depth = False

        """Decoders"""
        self.decoder_main_tasks = [["rgb", "depth"]]
        self.decoder_depth = 10

        # ConvNeXtAdapter
        self.dec_kernel = 3

        # debug
        # self.max_train_samples = 80
        # self.max_dev_samples = 80

        # Diversity of thought
        self.ground_truth_version = 6
        self.num_classes = 5  # ignored
        self.actual_num_classes = 5  # ignored
        self.additional_gt_tokens_mlp_channels = [768 * 2]

        """Learning rate"""
        self.lr = 1e-5
        self.end_lr = 1e-8
        self.lr_scale = 100

        self.batch_size = 20
        self.val_batch_size = 200
        self.nepochs = 400
        self.num_epochs_every_restart = 100

        self.data_augmentation_version = 6
        self.train_function_version = 3
        self.weight_decay = 5e-2
        self.transform1 = [
            A.HorizontalFlip(p=0.5),
        ]


class cfgv4_0_2007(base_cfg):
    def __init__(self, epoch: Optional[int] = None):
        super().__init__(epoch, datasets_set=1)

        self.check_val_every_n_epoch = 1
        self.num_workers = 4
        self.devices = [0]

        self.description = "ViT Base [DoT]"

        """MultiMAE"""
        self.pretrained_backbone = PRETRAINED_BACKBONE.S_MULTIMAE
        self.pretrained_backbone_version = "v2.0.1-pr"

        self.clip_grad = None
        self.normalized_depth = False

        """Decoders"""
        self.decoder_main_tasks = [["rgb", "depth"]]
        self.decoder_depth = 10

        # ConvNeXtAdapter
        self.dec_kernel = 3

        # debug
        # self.max_train_samples = 80
        # self.max_dev_samples = 80

        # Diversity of thought
        self.ground_truth_version = 6
        self.num_classes = 5  # ignored
        self.actual_num_classes = 5  # ignored
        self.additional_gt_tokens_mlp_channels = [768 * 2]

        """Learning rate"""
        self.lr = 1e-5
        self.end_lr = 1e-8
        self.lr_scale = 100

        self.batch_size = 40
        self.val_batch_size = 200
        self.nepochs = 400
        self.num_epochs_every_restart = 100

        self.data_augmentation_version = 6
        self.train_function_version = 3
        self.weight_decay = 5e-2
        self.transform1 = [
            A.HorizontalFlip(p=0.5),
            A.OneOf(
                [
                    A.Compose(
                        [
                            A.RandomCropFromBorders(
                                crop_left=0.3,
                                crop_right=0.3,
                                crop_top=0.3,
                                crop_bottom=0.3,
                                p=0.2,
                            ),
                            A.ShiftScaleRotate(
                                shift_limit=0.0625,
                                scale_limit=0.1,
                                rotate_limit=45,
                                p=0.1,
                            ),
                            A.Perspective(
                                p=0.2,
                                scale=(0.05, 0.1),
                            ),
                        ]
                    ),
                    A.Compose(
                        [
                            A.RandomCropFromBorders(
                                crop_left=0.3,
                                crop_right=0.3,
                                crop_top=0.3,
                                crop_bottom=0.3,
                                p=0.2,
                            ),
                            A.ShiftScaleRotate(
                                shift_limit=0.0625,
                                scale_limit=0.1,
                                rotate_limit=45,
                                p=0.1,
                                border_mode=cv2.BORDER_CONSTANT,
                                value=(255, 255, 255),
                                mask_value=0,
                            ),
                            A.Perspective(
                                p=0.2,
                                scale=(0.05, 0.1),
                                pad_mode=cv2.BORDER_CONSTANT,
                                pad_val=(255, 255, 255),
                                mask_pad_val=0,
                            ),
                        ]
                    ),
                ]
            ),
        ]


class cfgv4_0_2002(base_cfg):
    def __init__(self, epoch: Optional[int] = None):
        super().__init__(epoch, datasets_set=1)

        self.check_val_every_n_epoch = 1
        self.num_workers = 4
        self.devices = [3]

        # self.description = "Trainv3-DAv6-DiversityOfThought-NotMuchAug"
        self.description = "DEBUG"

        """MultiMAE"""
        self.pretrained_backbone = PRETRAINED_BACKBONE.S_MULTIMAE
        self.pretrained_backbone_version = "v2.0.1-pr"

        # Large MAE
        # self.dim_tokens = 1024
        # self.encoder_depth = 24
        # self.num_heads = 16

        self.clip_grad = None
        self.normalized_depth = False

        """Decoders"""
        self.decoder_main_tasks = [["rgb", "depth"]]
        self.decoder_depth = 10

        # ConvNeXtAdapter
        self.dec_kernel = 3

        # debug
        self.max_train_samples = 20
        self.max_dev_samples = 20

        # Diversity of thought
        self.ground_truth_version = 6
        self.num_classes = 5  # ignored
        self.actual_num_classes = 5  # ignored
        self.additional_gt_tokens_mlp_channels = [768 * 2]

        """Learning rate"""
        self.lr = 1e-5
        self.end_lr = 1e-8
        self.lr_scale = 100

        self.batch_size = 5
        self.val_batch_size = 5
        self.nepochs = 400
        self.num_epochs_every_restart = 100

        self.data_augmentation_version = 6
        self.train_function_version = 3
        self.weight_decay = 5e-2
        self.transform1 = [
            A.HorizontalFlip(p=0.5),
            A.OneOf(
                [
                    A.Compose(
                        [
                            A.RandomCropFromBorders(
                                crop_left=0.3,
                                crop_right=0.3,
                                crop_top=0.3,
                                crop_bottom=0.3,
                                p=0.2,
                            ),
                            A.ShiftScaleRotate(
                                shift_limit=0.0625,
                                scale_limit=0.1,
                                rotate_limit=45,
                                p=0.1,
                            ),
                            A.Perspective(
                                p=0.2,
                                scale=(0.05, 0.1),
                            ),
                        ]
                    ),
                    A.Compose(
                        [
                            A.RandomCropFromBorders(
                                crop_left=0.3,
                                crop_right=0.3,
                                crop_top=0.3,
                                crop_bottom=0.3,
                                p=0.2,
                            ),
                            A.ShiftScaleRotate(
                                shift_limit=0.0625,
                                scale_limit=0.1,
                                rotate_limit=45,
                                p=0.1,
                                border_mode=cv2.BORDER_CONSTANT,
                                value=(255, 255, 255),
                                mask_value=0,
                            ),
                            A.Perspective(
                                p=0.2,
                                scale=(0.05, 0.1),
                                pad_mode=cv2.BORDER_CONSTANT,
                                pad_val=(255, 255, 255),
                                mask_pad_val=0,
                            ),
                        ]
                    ),
                ]
            ),
        ]
