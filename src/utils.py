"""Shared helpers: config loading, seeding, device selection."""
import os
import random

import numpy as np
import torch
import yaml


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for sub in ["checkpoints", "figures", "metrics"]:
        os.makedirs(os.path.join(cfg["output_dir"], sub), exist_ok=True)
    return cfg


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ckpt_path(cfg: dict, model_name: str) -> str:
    return os.path.join(cfg["output_dir"], "checkpoints", f"{model_name}_best.pt")
