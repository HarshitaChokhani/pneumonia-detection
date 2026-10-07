"""Dataset loading, augmentation, stratified train/val split, class weights."""
import os

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(img_size: int):
    train_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    # No horizontal flip: heart position makes X-rays left/right asymmetric.
    eval_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


def get_dataloaders(cfg: dict):
    train_tf, eval_tf = get_transforms(cfg["img_size"])
    root = cfg["data_dir"]

    # Two views of the same folder so train and val get different transforms.
    full_train_aug = datasets.ImageFolder(os.path.join(root, "train"), transform=train_tf)
    full_train_eval = datasets.ImageFolder(os.path.join(root, "train"), transform=eval_tf)
    test_ds = datasets.ImageFolder(os.path.join(root, "test"), transform=eval_tf)

    targets = np.array(full_train_aug.targets)
    idx = np.arange(len(targets))
    train_idx, val_idx = train_test_split(
        idx, test_size=cfg["val_split"], stratify=targets, random_state=cfg["seed"]
    )
    train_ds = Subset(full_train_aug, train_idx)
    val_ds = Subset(full_train_eval, val_idx)

    kw = dict(batch_size=cfg["batch_size"], num_workers=cfg["num_workers"], pin_memory=True)
    loaders = {
        "train": DataLoader(train_ds, shuffle=True, **kw),
        "val": DataLoader(val_ds, shuffle=False, **kw),
        "test": DataLoader(test_ds, shuffle=False, **kw),
    }

    # Inverse-frequency class weights from the training split only.
    counts = np.bincount(targets[train_idx], minlength=len(full_train_aug.classes))
    weights = torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32)

    info = {
        "classes": full_train_aug.classes,
        "class_weights": weights,
        "sizes": {k: len(v.dataset) for k, v in loaders.items()},
        "train_counts": counts.tolist(),
        "test_counts": np.bincount(test_ds.targets).tolist(),
    }
    return loaders, info
