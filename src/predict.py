"""Predict a single X-ray from the command line.

Usage:  python -m src.predict path/to/xray.jpeg [--model resnet50]
"""
import argparse

import torch
from PIL import Image

from src.data import get_transforms
from src.models import get_model
from src.utils import ckpt_path, get_device, load_config


def load_model(cfg, name, device):
    model = get_model(name, len(cfg["classes"]), pretrained=False).to(device)
    model.load_state_dict(torch.load(ckpt_path(cfg, name), map_location=device))
    return model.eval()


@torch.no_grad()
def predict_image(model, img: Image.Image, cfg, device):
    _, eval_tf = get_transforms(cfg["img_size"])
    x = eval_tf(img.convert("RGB")).unsqueeze(0).to(device)
    probs = torch.softmax(model(x), 1)[0].cpu().tolist()
    return dict(zip(cfg["classes"], probs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--model", default="resnet50")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = get_device()
    probs = predict_image(load_model(cfg, args.model, device), Image.open(args.image), cfg, device)
    label = max(probs, key=probs.get)
    print(f"Prediction: {label} ({probs[label]:.2%})")
    for k, v in probs.items():
        print(f"  {k:<10} {v:.4f}")


if __name__ == "__main__":
    main()
