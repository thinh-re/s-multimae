from streamlit_apps.app_utils.model import base_inference
from streamlit_apps.app_utils.sod_selection_ui import load_smultimae_model
from PIL import Image
import numpy as np

d = {
    "S-MultiMAE [ViT-L] Multi-GT": {"top": 1, "cfg": "cfgv4_0_2006"},
    "S-MultiMAE [ViT-B] Multi-GT": {"top": 1, "cfg": "cfgv4_0_2007"},
}

sod_model_config_key = "S-MultiMAE [ViT-L] Multi-GT"
sod_model, cfg, da = load_smultimae_model(
    d[sod_model_config_key]["cfg"], d[sod_model_config_key]["top"]
)


color = np.array([0.0, 0.97647059, 0.0])
num_sets_of_salient_objects = 1


def infer(image: Image.Image, depth: Image.Image):
    pred_depth, pred_sods, pred_sms = base_inference(
        None,
        sod_model,
        da,
        image,
        depth,
        color,
        num_sets_of_salient_objects,
    )
    sod = Image.fromarray(pred_sods[0])
    return sod


if __name__ == "__main__":
    image = Image.open("data/inputs/1/rgb.jpg").convert("RGB")
    depth = Image.open("data/inputs/1/depth.jpg").convert("L")
    sod = infer(image, depth)
    sod.save("data/inputs/1/sod.jpg")
