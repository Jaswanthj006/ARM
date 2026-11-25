"""Fine-tune the PCB detector for any solder-mask colour (red, blue, black, yellow...).
The training tiles are green PKU boards; full hue augmentation teaches YOLO the defect shapes,
not the colour green. Starts from the current best weights; writes runs/pcb_yolo11n_colour/.
Usage: python3 ml/train_colour.py [--epochs 20]"""
import argparse
from pathlib import Path
import torch
from ultralytics import YOLO
ROOT = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=20); a = ap.parse_args()
YOLO(str(ROOT / "runs/pcb_yolo11n/weights/best.pt")).train(
    data=str(ROOT / "data/tiles/data.yaml"), imgsz=640, epochs=a.epochs, batch=16,
    device="mps" if torch.backends.mps.is_available() else "cpu", workers=4,
    project=str(ROOT / "runs"), name="pcb_yolo11n_colour", exist_ok=True, patience=10, cos_lr=True, seed=0,
    lr0=0.002, warmup_epochs=1,
    fliplr=0.5, flipud=0.5, degrees=0, mosaic=1.0, scale=0.3,
    hsv_h=0.5, hsv_s=0.6, hsv_v=0.4,          # any hue: green, red, blue, purple, yellow boards
)
