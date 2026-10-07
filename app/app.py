"""Gradio demo: upload a chest X-ray -> prediction + Grad-CAM heatmap.

Usage (from project root):  python -m app.app
"""
import matplotlib
matplotlib.use("Agg")
import gradio as gr
import matplotlib.cm as cm
import numpy as np

from src.gradcam import explain_image, load_cam
from src.utils import get_device, load_config

MODEL_NAME = "resnet50"
cfg = load_config()
device = get_device()
cam = load_cam(cfg, MODEL_NAME, device)


def analyze(img):
    if img is None:
        return None, None
    base, heat, _, probs = explain_image(cam, img, cfg["img_size"], device)
    overlay = 0.6 * base + 0.4 * cm.jet(heat)[..., :3]
    return {c: float(p) for c, p in zip(cfg["classes"], probs)}, np.clip(overlay, 0, 1)


demo = gr.Interface(
    fn=analyze,
    inputs=gr.Image(type="pil", label="Chest X-ray"),
    outputs=[gr.Label(num_top_classes=2, label="Prediction"),
             gr.Image(label="Grad-CAM (where the model looked)")],
    title="Pneumonia Detection from Chest X-rays",
    description="ResNet50 fine-tuned on the Kaggle Chest X-Ray dataset. "
                "Academic project, not for clinical use.",
)

if __name__ == "__main__":
    demo.launch(share=True)
