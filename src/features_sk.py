"""Approccio 3: feature congelate (output dopo avgpool, 576-d) + classificatori scikit-learn."""
import time

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import OxfordIIITPet

from src.data import ROOT, eval_tf, make_split
from src.models import build
from src.utils import set_seed, get_device, save_json

SEED = 42


@torch.no_grad()
def extract(model, loader, dev):
    """Feature a 576 dimensioni: features -> avgpool -> flatten (prima del classifier)."""
    model.eval()
    F, Y = [], []
    for x, y in loader:
        f = model.avgpool(model.features(x.to(dev))).flatten(1)
        F.append(f.cpu())
        Y.append(y)
    return torch.cat(F).numpy(), torch.cat(Y).numpy()


def main():
    set_seed(SEED)
    dev = get_device()
    tr, va, _, _ = make_split(SEED)
    # Nessuna augmentation: la rete è congelata, si usa eval_tf ovunque
    ds_tv = OxfordIIITPet(ROOT, "trainval", target_types="category", transform=eval_tf)
    ds_te = OxfordIIITPet(ROOT, "test", target_types="category", transform=eval_tf)
    mk = lambda ds: DataLoader(ds, 64, shuffle=False, num_workers=4)

    model = build("head").to(dev)      # stessi pesi pre-addestrati, usati solo come estrattore
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    Xtr, ytr = extract(model, mk(Subset(ds_tv, tr)), dev)
    Xva, yva = extract(model, mk(Subset(ds_tv, va)), dev)
    Xte, yte = extract(model, mk(ds_te), dev)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t_extract = time.perf_counter() - t0
    print(f"Feature: {Xtr.shape}, {Xva.shape}, {Xte.shape}, estrazione {t_extract:.1f}s")
    np.savez("results/features.npz", Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva, Xte=Xte, yte=yte)

    # Scelta dell'iperparametro sulla VALIDATION (scaler fittato sul solo train dentro la pipeline)
    grid = {
        "logreg": [(C, lambda C=C: LogisticRegression(C=C, max_iter=3000)) for C in (0.01, 0.1, 1, 10)],
        "linsvm": [(C, lambda C=C: LinearSVC(C=C, max_iter=20000)) for C in (0.001, 0.01, 0.1, 1)],
        "rf": [(n, lambda n=n: RandomForestClassifier(n_estimators=n, n_jobs=-1, random_state=SEED))
               for n in (200, 500)],
    }
    out = {"extract_time_s": t_extract, "seed": SEED, "models": {}}
    for name, cands in grid.items():
        best = (-1, None, None)
        for hp, mkclf in cands:
            pipe = make_pipeline(StandardScaler(), mkclf())
            pipe.fit(Xtr, ytr)
            acc = accuracy_score(yva, pipe.predict(Xva))
            print(f"{name} hp={hp} val_acc={acc:.4f}")
            if acc > best[0]:
                best = (acc, hp, mkclf)
        val_acc, hp, mkclf = best
        # Riaddestramento del solo candidato scelto, misurando il tempo di training
        pipe = make_pipeline(StandardScaler(), mkclf())
        t1 = time.perf_counter()
        pipe.fit(Xtr, ytr)
        t_fit = time.perf_counter() - t1
        train_acc = accuracy_score(ytr, pipe.predict(Xtr))
        out["models"][name] = {"best_hp": hp, "val_acc": val_acc, "train_acc": train_acc,
                               "fit_time_s": t_fit}
        print(f"==> {name}: hp={hp}, val={val_acc:.4f}, train={train_acc:.4f}, fit {t_fit:.1f}s")
    save_json(out, "results/features_selection.json")


if __name__ == "__main__":
    main()