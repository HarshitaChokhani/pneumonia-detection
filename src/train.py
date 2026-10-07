"""Train every model listed in the config, with early stopping on val loss.

Usage:  python -m src.train [--config configs/config.yaml] [--model resnet50]
"""
import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

from src.data import get_dataloaders
from src.models import count_params, get_model
from src.utils import ckpt_path, get_device, load_config, set_seed


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in tqdm(loader, leave=False, desc="train" if training else "val"):
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n


def plot_history(hist, name, out_dir):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ep = range(1, len(hist["train_loss"]) + 1)
    ax[0].plot(ep, hist["train_loss"], label="train"); ax[0].plot(ep, hist["val_loss"], label="val")
    ax[0].set_title(f"{name} - loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(ep, hist["train_acc"], label="train"); ax[1].plot(ep, hist["val_acc"], label="val")
    ax[1].set_title(f"{name} - accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{name}_training_curves.png"), dpi=150)
    plt.close(fig)


def train_model(name, cfg, loaders, info, device):
    print(f"\n===== Training {name} =====")
    model = get_model(name, len(info["classes"])).to(device)
    total, trainable = count_params(model)
    print(f"Params: {total:,} total | {trainable:,} trainable")

    weights = info["class_weights"].to(device) if cfg["use_class_weights"] else None
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["lr"][name], weight_decay=cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=1)

    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val, bad_epochs, start = float("inf"), 0, time.time()

    for epoch in range(1, cfg["epochs"][name] + 1):
        tl, ta = run_epoch(model, loaders["train"], criterion, device, optimizer)
        vl, va = run_epoch(model, loaders["val"], criterion, device)
        scheduler.step(vl)
        for k, v in zip(hist, [tl, vl, ta, va]):
            hist[k].append(v)
        print(f"Epoch {epoch:02d} | train loss {tl:.4f} acc {ta:.4f} | val loss {vl:.4f} acc {va:.4f}")

        if vl < best_val:
            best_val, bad_epochs = vl, 0
            torch.save(model.state_dict(), ckpt_path(cfg, name))
            print("  ✓ saved best checkpoint")
        else:
            bad_epochs += 1
            if bad_epochs >= cfg["early_stopping_patience"]:
                print("  Early stopping.")
                break

    minutes = (time.time() - start) / 60
    plot_history(hist, name, os.path.join(cfg["output_dir"], "figures"))
    summary = {"model": name, "total_params": total, "trainable_params": trainable,
               "epochs_run": len(hist["train_loss"]), "train_minutes": round(minutes, 2),
               "best_val_loss": best_val, "history": hist}
    with open(os.path.join(cfg["output_dir"], "metrics", f"{name}_train.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--model", default=None, help="train only this model")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device()
    print("Device:", device)

    loaders, info = get_dataloaders(cfg)
    print("Classes:", info["classes"], "| sizes:", info["sizes"],
          "| train counts:", info["train_counts"], "| class weights:", info["class_weights"].tolist())

    for name in ([args.model] if args.model else cfg["models"]):
        train_model(name, cfg, loaders, info, device)


if __name__ == "__main__":
    main()
