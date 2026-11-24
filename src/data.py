"""Dataset Oxford-IIIT Pet: split stratificato train/val e trasformazioni."""
import json
import os

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as T
from torchvision.datasets import OxfordIIITPet
from torchvision.models import MobileNet_V3_Small_Weights

ROOT = "data"
SPLIT_FILE = "splits/split.json"
VAL_FRAC = 0.15

_W = MobileNet_V3_Small_Weights.DEFAULT
_tf = _W.transforms()                    # resize 256 -> center crop 224 -> normalize ImageNet
MEAN, STD = _tf.mean, _tf.std

# Augmentation SOLO per il training
train_tf = T.Compose([
    T.RandomResizedCrop(224, scale=(0.7, 1.0)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.2, 0.2, 0.2),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])
eval_tf = _tf


def make_split(seed: int = 42):
    """Crea (o carica) lo split stratificato di trainval e lo salva in splits/split.json."""
    base = OxfordIIITPet(ROOT, split="trainval", target_types="category", download=True)
    labels = np.array(base._labels)
    if os.path.exists(SPLIT_FILE):
        s = json.load(open(SPLIT_FILE))
        return np.array(s["train"]), np.array(s["val"]), labels, base.classes
    idx = np.arange(len(labels))
    tr, va = train_test_split(idx, test_size=VAL_FRAC, stratify=labels, random_state=seed)
    os.makedirs("splits", exist_ok=True)
    json.dump({"seed": seed, "train": tr.tolist(), "val": va.tolist()}, open(SPLIT_FILE, "w"))
    return tr, va, labels, base.classes


def get_loaders(bs: int = 64, num_workers: int = 4, seed: int = 42):
    """Restituisce train/val/test loader, classi e info sui conteggi."""
    tr, va, labels, classes = make_split(seed)
    ds_tr = OxfordIIITPet(ROOT, "trainval", target_types="category", transform=train_tf)
    ds_va = OxfordIIITPet(ROOT, "trainval", target_types="category", transform=eval_tf)
    ds_te = OxfordIIITPet(ROOT, "test", target_types="category", transform=eval_tf)

    import torch
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(Subset(ds_tr, tr), bs, shuffle=True, num_workers=num_workers, generator=g)
    val_loader = DataLoader(Subset(ds_va, va), bs, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(ds_te, bs, shuffle=False, num_workers=num_workers)

    test_labels = np.array(ds_te._labels)
    counts = {
        "train": np.bincount(labels[tr], minlength=37).tolist(),
        "val": np.bincount(labels[va], minlength=37).tolist(),
        "test": np.bincount(test_labels, minlength=37).tolist(),
    }
    return train_loader, val_loader, test_loader, classes, counts