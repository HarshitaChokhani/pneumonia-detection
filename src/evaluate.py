"""Evaluate trained models on the held-out test set.

Outputs: metrics JSON + comparison CSV, confusion matrices, ROC curves.
Usage:  python -m src.evaluate
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score, roc_curve)

from src.data import get_dataloaders
from src.models import get_model
from src.utils import ckpt_path, get_device, load_config, set_seed


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        p = torch.softmax(model(x.to(device)), dim=1)[:, 1]  # P(PNEUMONIA)
        probs.append(p.cpu().numpy()); labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def compute_metrics(y, prob, thr=0.5):
    pred = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall_sensitivity": recall_score(y, pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, prob) if len(set(y)) > 1 else float("nan"),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def plot_cm(y, prob, classes, name, out_dir):
    cm = confusion_matrix(y, (prob >= 0.5).astype(int), labels=[0, 1])
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, cbar=False)
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(f"{name} - confusion matrix")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"{name}_confusion_matrix.png"), dpi=150); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device()
    loaders, info = get_dataloaders(cfg)
    fig_dir = os.path.join(cfg["output_dir"], "figures")
    met_dir = os.path.join(cfg["output_dir"], "metrics")

    rows, roc_data = [], {}
    for name in cfg["models"]:
        path = ckpt_path(cfg, name)
        if not os.path.exists(path):
            print(f"Skipping {name}: no checkpoint at {path}"); continue
        model = get_model(name, len(info["classes"]), pretrained=False).to(device)
        model.load_state_dict(torch.load(path, map_location=device))

        prob, y = predict(model, loaders["test"], device)
        m = compute_metrics(y, prob)
        rows.append({"model": name, **m})
        roc_data[name] = (y, prob)
        plot_cm(y, prob, info["classes"], name, fig_dir)

        print(f"\n===== {name} (test set) =====")
        print(classification_report(y, (prob >= 0.5).astype(int), target_names=info["classes"], digits=4))
        with open(os.path.join(met_dir, f"{name}_test.json"), "w") as f:
            json.dump(m, f, indent=2)

    if not rows:
        print("No trained models found. Run `python -m src.train` first."); return

    # ROC comparison
    plt.figure(figsize=(5.5, 5))
    for name, (y, prob) in roc_data.items():
        fpr, tpr, _ = roc_curve(y, prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y, prob):.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate (sensitivity)")
    plt.title("ROC curve - test set"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "roc_comparison.png"), dpi=150); plt.close()

    df = pd.DataFrame(rows).set_index("model")
    df.round(4).to_csv(os.path.join(met_dir, "comparison.csv"))
    print("\n===== Model comparison =====")
    print(df[["accuracy", "precision", "recall_sensitivity", "specificity", "f1", "roc_auc"]].round(4))


if __name__ == "__main__":
    main()
