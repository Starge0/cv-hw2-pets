"""Esempi rappresentativi: un errore per ciascuna delle 3 coppie più confuse
(riga superiore) + una predizione corretta della stessa classe vera (riga inferiore)."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from src.data import get_loaders, MEAN, STD
from src.evaluate import NN, predict, top_confused
from src.models import build
from src.utils import get_device, set_seed


def main():
    set_seed(42)
    dev = get_device()
    _, _, test_loader, classes, _ = get_loaders(bs=64, seed=42)
    ds = test_loader.dataset
    idx = {c: i for i, c in enumerate(classes)}
    mean, std = torch.tensor(MEAN).view(3, 1, 1), torch.tensor(STD).view(3, 1, 1)

    for mode, jpath in NN.items():
        info = json.load(open(jpath))
        model = build(mode).to(dev)
        model.load_state_dict(torch.load(info["checkpoint"], map_location=dev))
        p, y = predict(model, test_loader, dev)
        cm = confusion_matrix(y, p, labels=range(37))
        cells = []
        for t, pr, n in top_confused(cm, classes, k=3):
            ti, pi = idx[t], idx[pr]
            err = int(np.where((y == ti) & (p == pi))[0][0])
            ok = int(np.where((y == ti) & (p == ti))[0][0])
            cells.append((err, ok, n))

        fig, axs = plt.subplots(2, 3, figsize=(10, 7))
        for col, (err, ok, n) in enumerate(cells):
            for row, i in enumerate((err, ok)):
                x, _ = ds[i]
                a = axs[row, col]
                a.imshow((x * std + mean).clamp(0, 1).permute(1, 2, 0).numpy())
                a.set_title(f"vero: {classes[y[i]]}\npred: {classes[p[i]]}"
                            + (f"  ({n} err.)" if row == 0 else ""),
                            fontsize=8, color="red" if row == 0 else "green")
                a.axis("off")
        fig.tight_layout()
        fig.savefig(f"results/examples_confused_{mode}.png", dpi=130)
        plt.close(fig)
        print(mode, "ok")


if __name__ == "__main__":
    main()