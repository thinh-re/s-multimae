import time
import numpy as np
import streamlit as st
from PIL import Image

from s_multimae.da.base_da import BaseDataAugmentation
from .base_model import BaseRGBDModel
from .depth_model import BaseDepthModel
from .model import base_inference


def image_inference(
    depth_model: BaseDepthModel,
    sod_model: BaseRGBDModel,
    da: BaseDataAugmentation,
    color: np.ndarray,
) -> None:
    if "depth" not in st.session_state:
        st.session_state.depth = None

    col1, col2 = st.columns(2)
    image: Image = None
    # depth: Image = None

    def file_uploader_on_change():
        st.session_state.depth = None

    with col1:
        img_file_buffer = st.file_uploader(
            "Upload an RGB image",
            key="img_file_buffer",
            type=["png", "jpg", "jpeg"],
            on_change=file_uploader_on_change,
        )
        if img_file_buffer is not None:
            image = Image.open(img_file_buffer).convert("RGB")
            st.image(image, caption="RGB")

    with col2:
        depth_file_buffer = st.file_uploader(
            "Upload a depth image (Optional)",
            key="depth_file_buffer",
            type=["png", "jpg", "jpeg"],
        )
        if depth_file_buffer is not None:
            st.session_state.depth = Image.open(depth_file_buffer).convert("L")
        if st.session_state.depth is not None:
            st.image(st.session_state.depth, caption="Depth")

    if sod_model.cfg.ground_truth_version == 6:
        num_sets_of_salient_objects = st.number_input(
            "Number of sets of salient objects", value=1, min_value=1, max_value=10
        )
    else:
        num_sets_of_salient_objects = 1

    is_predict = st.button(
        "Predict Salient Objects",
        key="predict_salient_objects",
        disabled=img_file_buffer is None,
    )
    if is_predict:
        with st.spinner("Processing... (it takes about 1-2 minutes)"):
            start_time = time.time()
            pred_depth, pred_sods, pred_sms = base_inference(
                depth_model,
                sod_model,
                da,
                image,
                st.session_state.depth,
                color,
                num_sets_of_salient_objects,
            )
            if st.session_state.depth is None:
                st.session_state.depth = Image.fromarray(pred_depth).convert("L")
                col2.image(st.session_state.depth, "Pseudo-depth")

            if num_sets_of_salient_objects == 1:
                st.warning(
                    "HINT: To view a wider variety of sets of salient objects, try to increase the number of sets the model can produce."
                )
            elif num_sets_of_salient_objects > 1:
                st.warning(
                    "NOTE: As single-GT accounts for 77.61% of training samples, the model may not consistently yield different sets. The best approach is to gradually increase the number of sets of salient objects until you achieve the desired result."
                )

            st.info(f"Inference time: {time.time() - start_time:.4f} seconds")

            sod_cols = st.columns(len(pred_sods))

            for i, (pred_sod, pred_sm) in enumerate(zip(pred_sods, pred_sms)):
                with sod_cols[i]:
                    st.image(pred_sod, "Salient Objects (Otsu threshold)")
                    st.image(pred_sm, "Salient Map")
