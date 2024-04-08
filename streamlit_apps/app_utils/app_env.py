import os

app_env = os.environ.get("APP_ENVIRONMENT", "HUGGINGFACE")

IMAGE_SIZE = 224


class DEPTH_MODEL_TYPE:
    DPT_DEPTH = "DPTDepth"
    REL_DEPTH = "RelDepth"


class SOD_MODEL_TYPE:
    S_MULTIMAE = "S-MultiMAE"
    SPNET = "SPNet"
    BBSNET = "BBSNet"
