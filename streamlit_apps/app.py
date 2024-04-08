import os, sys

sys.path.append(os.getcwd())

import multiprocessing

import streamlit as st

from app_utils.color_selection_ui import color_selection_ui
from app_utils.depth_selection_ui import depth_selection_ui
from app_utils.device import device
from app_utils.sod_selection_ui import sod_selection_ui


class MODE:
    IMAGE = "image"
    VIDEO = "video"
    WEBRTC = "webrtc"
    DEMO = "demo"


TITLE = "S-MultiMAE: A Multi-Ground Truth approach for RGB-D Saliency Detection"

st.set_page_config(
    page_title=TITLE,
    page_icon="🧊",
    layout="wide",
    # initial_sidebar_state="expanded",
    # menu_items={
    #     'Get Help': 'https://www.extremelycoolapp.com/help',
    #     'Report a bug': "https://www.extremelycoolapp.com/bug",
    #     'About': "# This is a header. This is an *extremely* cool app!"
    # }
)
st.title(TITLE)

with st.expander("INTRODUCTION"):
    st.text(
        f"""Demo for S-MultiMAE. 
        Device: {device.type}
        Number of CPU(s): {multiprocessing.cpu_count()}"""
    )
    st.image("docs/figures/proposed_method_v5.drawio.png", use_column_width="always")

with st.expander("SETTINGS", expanded=True):
    col1, col2 = st.columns(2)

    with col1:
        mode = st.radio(
            "Mode",
            (
                MODE.IMAGE,
                # MODE.VIDEO,
                # MODE.WEBRTC,
                # MODE.DEMO,
            ),
        )
        st.markdown("---")
        color = color_selection_ui()

    with col2:
        depth_model = depth_selection_ui()
        st.markdown("---")
        sod_model, da = sod_selection_ui()

with st.expander("HOW TO USE", expanded=True):
    st.text(
        "(1) You can change the model type (using different backbones) in the settings."
    )
    st.text("(2) Upload an RGB image.")
    st.text(
        "(3) (Optional) Provide its corresponding depth. If not present, a pseudo-depth will be inferred by a rgb2depth model."
    )
    st.text(
        "(4) You may try a different number of sets of salient objects the model can produce."
    )
    st.text("""(5) Click "Predict Salient Objects".""")

if mode == MODE.IMAGE:
    from app_utils.image_inference import image_inference

    image_inference(depth_model, sod_model, da, color)
# elif mode == MODE.VIDEO:
#     from video_inference import video_inference
#     video_inference(depth_model, sod_model, color)
# elif mode == MODE.WEBRTC:
#     from webrtc_app import webrtc_app
#     webrtc_app(depth_model, sod_model, color)
# elif mode == MODE.DEMO:
#     from demo import demo
#     demo()
