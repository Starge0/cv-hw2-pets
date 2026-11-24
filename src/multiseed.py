"""Media ± std su più seed per full e head (iperparametri già fissati sulla validation)."""
import json

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from src.data import get_loaders
from src.evaluate import predict
from src.models import build
from src.utils import get_device, save_json

CFG = {"full": 3e-4, "head": 1e-3}
SEEDS = (42, 1, 2)


def main():
    dev = get_device()
    _, _, test_loader, _, _ = get_loaders(bs=64, seed=42)
    out = {}
    for mode, lr in CFG.items():
        accs, f1s, times = [], [], []
        for s in SEEDS:
            info = json.load(open(f"results/{mode}_lr{lr}_s{s}_train.json"))
            model = build(mode).to(dev)
            model.load_state_dict(torch.load(info["checkpoint"], map_location=dev))
            p, y = predict(model, test_loader, dev)
            accs.append(accuracy_score(y, p))
            f1s.append(f1_score(y, p, average="macro"))
            times.append(info["train_time_s"])
            print(mode, s, round(accs[-1], 4), round(f1s[-1], 4))
        out[mode] = {"seeds": list(SEEDS), "test_acc": accs, "macro_f1": f1s, "time_s": times,
                     "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs, ddof=1)),
                     "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s, ddof=1))}
        print(f"==> {mode}: acc {np.mean(accs):.4f} ± {np.std(accs, ddof=1):.4f}, "
              f"F1 {np.mean(f1s):.4f} ± {np.std(f1s, ddof=1):.4f}")
    save_json(out, "results/multiseed.json")


if __name__ == "__main__":
    main()