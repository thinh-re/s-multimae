"""
Do not import other modules!
"""


class PRETRAINED_BACKBONE:
    MULTIMAE = "multimae"

    S_MULTIMAE = "s-multimae"
    LARGE_S_MULTIMAE = "large-s-multimae"

    MAE = "mae"
    LARGE_MAE = "large-mae"
    HUGE_MAE = "huge-mae"

    FINETUNE_LARGE_S_MULTIMAE = "finetune-large-s-multimae"
    FINETUNE_S_MULTIMAE = "finetune-s-multimae"

    VIT = "vit"  # train from supervised model

    NONE = None  # train from scratch
