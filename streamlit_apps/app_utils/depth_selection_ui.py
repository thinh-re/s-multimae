import streamlit as st

from .app_env import DEPTH_MODEL_TYPE, IMAGE_SIZE
from .depth_model import BaseDepthModel, DPTDepth


@st.cache_resource
def load_depth_model(depth_model_type: DEPTH_MODEL_TYPE) -> DPTDepth:
    if depth_model_type == DEPTH_MODEL_TYPE.DPT_DEPTH:
        return DPTDepth(IMAGE_SIZE)
    else:
        return DPTDepth(IMAGE_SIZE)


def depth_selection_ui() -> BaseDepthModel:
    depth_model: BaseDepthModel = None
    depth_model_type = st.selectbox(
        "Choose depth model",
        (
            DEPTH_MODEL_TYPE.DPT_DEPTH,
            # DEPTH_MODEL_TYPE.REL_DEPTH,
        ),
        key="depth_model_type",
    )
    depth_model = load_depth_model(depth_model_type)
    st.text(f"Number of parameters {depth_model.get_number_of_parameters()}")
    return depth_model
