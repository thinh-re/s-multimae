from functools import partial
from typing import Dict, Optional, Type

from .base_config import base_cfg
import importlib, inspect, os
from glob import glob

arg_cfg: Dict[str, Type[base_cfg]] = dict()

modules = []
for p in glob("s_multimae/configs/experiment_configs/*.py"):
    if not p.startswith("__"):
        module_name = os.path.splitext(os.path.basename(p))[0]
        modules.append(f"s_multimae.configs.experiment_configs.{module_name}")

for module in modules:
    for name, cls in inspect.getmembers(
        importlib.import_module(module), inspect.isclass
    ):
        if name.startswith("cfg"):
            arg_cfg[name] = cls


def get_config_by_set_version(set_version: int) -> base_cfg:
    if set_version not in [1, 2, 3, 4]:
        raise Exception(f"Unsupported set version {set_version}")
    return arg_cfg[f"cfg_set_{set_version}"]()


def get_config(cfg_name: str, epoch: Optional[int] = None) -> base_cfg:
    return arg_cfg[cfg_name](epoch)
