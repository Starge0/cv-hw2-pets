"""Valutazione finale sul TEST (una sola volta): metriche, matrici di confusione, esempi."""
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.data import get_loaders, MEAN, STD
from src.models import build, n_trainable
from src.utils import get_device, save_json, set_seed

SEED = 42
# Scelte fatte SOLO sulla validation (vedi results/*_train.json e results/features_selection.json)
NN = {"full": "results/full_lr0.0003_s42_train.json", "head": "results/head_lr0.001_s42_train.json"}


@torch.no_grad()
def predict(model, loader, dev):
    model.eval()
    P, Y = [], []
    for x, y in loader:
        P.append(model(x.to(dev)).argmax(1).cpu())
        Y.append(y)
    return torch.cat(P).numpy(), torch.cat(Y).numpy()


def plot_cm(cm, classes, title, path):
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(16, 16))
    ax.imshow(cmn, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=90, fontsize=8)
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel("Predetto")
    ax.set_ylabel("Vero")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def top_confused(cm, classes, k=5):
    c = cm.copy()
    np.fill_diagonal(c, 0)
    idx = np.dstack(np.unravel_index(np.argsort(-c, axis=None)[:k], c.shape))[0]
    return [(classes[i], classes[j], int(c[i, j])) for i, j in idx]


def show_examples(imgs_loader, y_true, y_pred, classes, path):
    """Griglia di 6 corrette + 6 sbagliate (immagini de-normalizzate)."""
    ok = np.where(y_true == y_pred)[0][:6]
    ko = np.where(y_true != y_pred)[0][:6]
    sel = list(ok) + list(ko)
    ds = imgs_loader.dataset
    mean, std = torch.tensor(MEAN).view(3, 1, 1), torch.tensor(STD).view(3, 1, 1)
    fig, axs = plt.subplots(2, 6, figsize=(18, 6.5))
    for a, i in zip(axs.ravel(), sel):
        x, _ = ds[int(i)]
        a.imshow((x * std + mean).clamp(0, 1).permute(1, 2, 0).numpy())
        a.set_title(f"vero: {classes[y_true[i]]}\npred: {classes[y_pred[i]]}", fontsize=8,
                    color="green" if y_true[i] == y_pred[i] else "red")
        a.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    set_seed(SEED)
    dev = get_device()
    _, _, test_loader, classes, _ = get_loaders(bs=64, seed=SEED)
    results, preds = {}, {}

    # Approcci 1 e 2: checkpoint scelto sulla validation
    for mode, jpath in NN.items():
        info = json.load(open(jpath))
        model = build(mode).to(dev)
        model.load_state_dict(torch.load(info["checkpoint"], map_location=dev))
        p, y = predict(model, test_loader, dev)
        h = info["history"][info["best_epoch"] - 1]
        results[mode] = {"trainable_params": info["trainable_params"], "time_s": info["train_time_s"],
                         "train_acc": h["train_acc"], "val_acc": info["best_val_acc"],
                         "test_acc": accuracy_score(y, p), "macro_f1": f1_score(y, p, average="macro"),
                         "lr": info["lr"], "best_epoch": info["best_epoch"]}
        preds[mode] = p
        y_test = y

    # Approccio 3: classificatori sklearn con iperparametri scelti sulla validation
    d = np.load("results/features.npz")
    sel = json.load(open("results/features_selection.json"))
    mk = {"logreg": lambda hp: LogisticRegression(C=hp, max_iter=3000),
          "linsvm": lambda hp: LinearSVC(C=hp, max_iter=20000),
          "rf": lambda hp: RandomForestClassifier(n_estimators=hp, n_jobs=-1, random_state=SEED)}
    nfeat = d["Xtr"].shape[1]
    for name, m in sel["models"].items():
        pipe = make_pipeline(StandardScaler(), mk[name](m["best_hp"]))
        pipe.fit(d["Xtr"], d["ytr"])                 # scaler e classificatore su solo train
        p = pipe.predict(d["Xte"])
        results[name] = {"trainable_params": nfeat * 37 + 37 if name != "rf" else None,
                         "time_s": sel["extract_time_s"] + m["fit_time_s"],
                         "train_acc": m["train_acc"], "val_acc": m["val_acc"],
                         "test_acc": accuracy_score(d["yte"], p),
                         "macro_f1": f1_score(d["yte"], p, average="macro"), "hp": m["best_hp"]}
        preds[name] = p

    # Matrici di confusione, coppie confuse, esempi
    conf = {}
    for k, p in preds.items():
        cm = confusion_matrix(y_test, p, labels=range(37))
        plot_cm(cm, classes, f"Matrice di confusione (test): {k}", f"results/cm_{k}.png")
        conf[k] = top_confused(cm, classes)
    show_examples(test_loader, y_test, preds["full"], classes, "results/examples_full.png")
    show_examples(test_loader, y_test, preds["head"], classes, "results/examples_head.png")

    save_json({"metrics": results, "top_confused": conf}, "results/test_metrics.json")
    print(f"{'metodo':8s} {'test_acc':>8s} {'macro_f1':>8s} {'train_acc':>9s} {'tempo_s':>8s}")
    for k, r in results.items():
        print(f"{k:8s} {r['test_acc']:8.4f} {r['macro_f1']:8.4f} {r['train_acc']:9.4f} {r['time_s']:8.1f}")
    for k, v in conf.items():
        print(k, v)


if __name__ == "__main__":
    main()