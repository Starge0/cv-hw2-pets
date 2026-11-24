"""Utility: seed, device, timer, salvataggio JSON."""
import json
import random
import time
from contextlib import contextmanager

import numpy as np
import torch

SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Fissa i seed di Python, NumPy e PyTorch e abilita le operazioni deterministiche."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@contextmanager
def timer(device: torch.device):
    """Misura il tempo reale (con sincronizzazione GPU). Uso: with timer(dev) as t: ... ; t['s']"""
    res = {}
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    yield res
    if device.type == "cuda":
        torch.cuda.synchronize()
    res["s"] = time.perf_counter() - t0


def save_json(obj, path: str) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)