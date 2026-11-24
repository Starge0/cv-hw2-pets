"""Training di full fine-tuning e head-only. Seleziona il checkpoint sulla validation."""
import argparse
import time

import torch
import torch.nn as nn

from src.data import get_loaders
from src.models import build, n_trainable
from src.utils import set_seed, get_device, save_json


@torch.no_grad()
def evaluate(model, loader, dev, crit):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        out = model(x)
        loss_sum += crit(out, y).item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / total, loss_sum / total


def run(mode, lr, epochs, bs, seed, wd=1e-4):
    set_seed(seed)
    dev = get_device()
    train_loader, val_loader, _, _, _ = get_loaders(bs=bs, seed=seed)
    model = build(mode).to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    crit = nn.CrossEntropyLoss()

    tag = f"{mode}_lr{lr}_s{seed}"
    ckpt = f"results/{tag}_best.pt"
    best_val, best_ep, history = 0.0, -1, []

    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for ep in range(epochs):
        model.train()
        if mode == "head":
            model.features.eval()      # BatchNorm congelate (niente update delle statistiche)
        run_loss, run_corr, n = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            run_loss += loss.item() * y.size(0)
            run_corr += (out.argmax(1) == y).sum().item()
            n += y.size(0)
        val_acc, val_loss = evaluate(model, val_loader, dev, crit)
        history.append({"epoch": ep + 1, "train_loss": run_loss / n, "train_acc": run_corr / n,
                        "val_loss": val_loss, "val_acc": val_acc})
        print(f"[{tag}] ep {ep+1}/{epochs} train_acc={run_corr/n:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_val:                       # selezione SOLO su validation
            best_val, best_ep = val_acc, ep + 1
            torch.save(model.state_dict(), ckpt)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    train_time = time.perf_counter() - t0

    save_json({"mode": mode, "lr": lr, "epochs": epochs, "batch_size": bs, "seed": seed,
               "weight_decay": wd, "optimizer": "AdamW", "loss": "CrossEntropy",
               "trainable_params": n_trainable(model), "train_time_s": train_time,
               "best_val_acc": best_val, "best_epoch": best_ep, "checkpoint": ckpt,
               "history": history}, f"results/{tag}_train.json")
    print(f"Fatto: best val {best_val:.4f} @ epoch {best_ep}, tempo {train_time:.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["full", "head"], required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    run(a.mode, a.lr, a.epochs, a.bs, a.seed)