"""Grad-CAM: highlights the image regions that drove the model's prediction.

Usage:  python -m src.gradcam --model resnet50 --n 8
"""
import argparse
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.data import IMAGENET_MEAN, IMAGENET_STD, get_transforms
from src.models import get_model, gradcam_target_layer
from src.utils import ckpt_path, get_device, load_config


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.acts, self.grads = None, None
        target_layer.register_forward_hook(lambda m, i, o: setattr(self, "acts", o.detach()))
        target_layer.register_full_backward_hook(lambda m, gi, go: setattr(self, "grads", go[0].detach()))

    def __call__(self, x, class_idx=None):
        x = x.clone().requires_grad_(True)
        out = self.model(x)
        probs = torch.softmax(out, 1)[0]
        if class_idx is None:
            class_idx = int(out.argmax(1))
        self.model.zero_grad()
        out[0, class_idx].backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)          # GAP over gradients
        cam = F.relu((w * self.acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.cpu().numpy(), class_idx, probs.detach().cpu().numpy()


def denormalize(t):
    img = t.cpu().numpy().transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img, 0, 1)


def load_cam(cfg, name, device):
    model = get_model(name, len(cfg["classes"]), pretrained=False).to(device)
    model.load_state_dict(torch.load(ckpt_path(cfg, name), map_location=device))
    return GradCAM(model, gradcam_target_layer(model, name))


def explain_image(cam, img: Image.Image, img_size, device):
    _, eval_tf = get_transforms(img_size)
    x = eval_tf(img.convert("RGB")).unsqueeze(0).to(device)
    heat, cls, probs = cam(x)
    return denormalize(x[0]), heat, cls, probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--model", default="resnet50")
    ap.add_argument("--n", type=int, default=8, help="number of test images (half per class)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = get_device()
    cam = load_cam(cfg, args.model, device)
    random.seed(cfg["seed"])

    test_dir = os.path.join(cfg["data_dir"], "test")
    samples = []
    for label, cls in enumerate(cfg["classes"]):
        files = sorted(os.listdir(os.path.join(test_dir, cls)))
        samples += [(os.path.join(test_dir, cls, f), label) for f in random.sample(files, args.n // 2)]

    fig, axes = plt.subplots(2, len(samples), figsize=(2.6 * len(samples), 5.6))
    for i, (path, label) in enumerate(samples):
        img, heat, pred, probs = explain_image(cam, Image.open(path), cfg["img_size"], device)
        axes[0, i].imshow(img); axes[0, i].axis("off")
        axes[0, i].set_title(f"True: {cfg['classes'][label]}", fontsize=9)
        axes[1, i].imshow(img); axes[1, i].imshow(heat, cmap="jet", alpha=0.4); axes[1, i].axis("off")
        color = "green" if pred == label else "red"
        axes[1, i].set_title(f"Pred: {cfg['classes'][pred]} ({probs[pred]:.2f})", fontsize=9, color=color)
    fig.suptitle(f"Grad-CAM - {args.model}")
    fig.tight_layout()
    out = os.path.join(cfg["output_dir"], "figures", f"{args.model}_gradcam.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print("Saved", out)


if __name__ == "__main__":
    main()
