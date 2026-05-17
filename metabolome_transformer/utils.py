import json
from pathlib import Path

import numpy as np
import torch


def choose_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_json(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_lines(path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def clamp_nonnegative(values):
    return np.maximum(values, 0.0)


def ensure_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
