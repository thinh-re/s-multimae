import numpy as np
import streamlit as st

from s_multimae.utils import hex_to_rgb


def color_selection_ui() -> np.ndarray:
    color = st.color_picker("Pick A Color", value="#00f900", key="color")
    color = hex_to_rgb(color)
    print("color", color)
    return color
