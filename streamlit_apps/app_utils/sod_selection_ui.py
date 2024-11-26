from typing import Tuple
import streamlit as st
import os
import torch

from .app_env import SOD_MODEL_TYPE
from .app_utils import count_parameters
from .smultimae_model import RGBDSMultiMAEModel
from .base_model import BaseRGBDModel
from .device import device

from s_multimae.da.dav6 import DataAugmentationV6
from s_multimae.configs.base_config import base_cfg
from s_multimae.configs.experiment_config import arg_cfg
from s_multimae.model_pl import ModelPL

# from spnet_model import SPNetModel


@st.cache_resource
def load_smultimae_model(
    sod_model_config_key: str, top: int
) -> Tuple[BaseRGBDModel, base_cfg]:
    """
    1. Construct model
    2. Load pretrained weights
    3. Load model into device
    """
    cfg = arg_cfg[sod_model_config_key]()

    weights_fname = f"s-multimae-{cfg.experiment_name}-top{top}.pth"
    ckpt_path = os.path.join("weights", weights_fname)
    print(ckpt_path)
    if not os.path.isfile(ckpt_path):
        from huggingface_hub import hf_hub_download

        hf_hub_download(
            repo_id="RGBD-SOD/S-MultiMAE",
            filename=weights_fname,
            local_dir="weights",
        )
    assert os.path.isfile(ckpt_path)

    # sod_model = ModelPL.load_from_checkpoint(
    #     ckpt_path,
    #     cfg=cfg,
    #     map_location=device,
    # )
    sod_model = ModelPL(cfg)
    sod_model.model.load_state_dict(
        torch.load(ckpt_path, map_location=device), strict=False
    )
    da = DataAugmentationV6(cfg)
    return RGBDSMultiMAEModel(cfg, sod_model), cfg, da


# @st.cache_resource
# def load_spnet_model() -> BaseRGBDModel:
#     """
#     1. Construct model
#     2. Load pretrained weights
#     3. Load model into device
#     """
#     sod_model = SPNetModel()
#     return sod_model


# @st.cache_resource
# def load_bbsnet_model() -> BaseRGBDModel:
#     """
#     1. Construct model
#     2. Load pretrained weights
#     3. Load model into device
#     """
#     sod_model = BBSNetModel()
#     return sod_model


def sod_selection_ui() -> BaseRGBDModel:
    sod_model_type = st.selectbox(
        "Choose SOD model",
        (
            SOD_MODEL_TYPE.S_MULTIMAE,
            # SOD_MODEL_TYPE.SPNET,
            # SOD_MODEL_TYPE.BBSNET,
        ),
        key="sod_model_type",
    )

    if sod_model_type == SOD_MODEL_TYPE.S_MULTIMAE:
        d = {
            "S-MultiMAE [ViT-L] Multi-GT": {"top": 1, "cfg": "cfgv4_0_2006"},
            "S-MultiMAE [ViT-B] Multi-GT": {"top": 1, "cfg": "cfgv4_0_2007"},
        }

        sod_model_config_key = st.selectbox(
            "Choose config",
            list(d.keys()),
            key="sod_model_config_key",
        )
        sod_model, cfg, da = load_smultimae_model(
            d[sod_model_config_key]["cfg"], d[sod_model_config_key]["top"]
        )
        # st.text(f"Model description: {cfg.description}")
    # elif sod_model_type == SOD_MODEL_TYPE.SPNET:
    #     sod_model = load_spnet_model()
    #     st.text(f"Model description: SPNet (https://github.com/taozh2017/SPNet)")
    # elif sod_model_type == SOD_MODEL_TYPE.BBSNET:
    #     sod_model = load_bbsnet_model()
    #     st.text(f"Model description: BBSNet (https://github.com/DengPingFan/BBS-Net)")
    st.text(f"Number of parameters {count_parameters(sod_model)}")

    return sod_model, da
