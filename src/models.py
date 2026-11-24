"""Costruzione di MobileNetV3-Small e gestione dei parametri allenabili."""
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

NUM_CLASSES = 37


def build(mode: str) -> nn.Module:
    """mode: 'full' (tutta la rete allenabile) oppure 'head' (solo classifier)."""
    m = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    # classifier = Linear(576,1024) -> Hardswish -> Dropout -> Linear(1024,1000)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, NUM_CLASSES)
    if mode == "head":
        for p in m.features.parameters():   # congela l'estrattore convoluzionale
            p.requires_grad = False
    return m


def n_trainable(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def n_total(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())