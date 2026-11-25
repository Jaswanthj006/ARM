"""Train the PCB defect detector (YOLO11 nano) on 640 px tiles.

Runs on the Mac GPU (Apple MPS) when available, else CPU.
Usage:  python3 ml/train.py [--epochs 60] [--model yolo11n.pt]
"""
import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--resume", action="store_true", help="continue from runs/pcb_yolo11n/weights/last.pt")
    a = ap.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if a.resume:
        YOLO(str(ROOT / "runs/pcb_yolo11n/weights/last.pt")).train(resume=True, device=device)
        return
    model = YOLO(a.model)
    model.train(
        data=str(ROOT / "data" / "tiles" / "data.yaml"),
        imgsz=640, epochs=a.epochs, batch=a.batch, device=device, workers=4,
        project=str(ROOT / "runs"), name="pcb_yolo11n", exist_ok=True,
        patience=20, cos_lr=True, seed=0,
        # PCB images: colour and orientation matter little, defects are tiny.
        fliplr=0.5, flipud=0.5, degrees=0, mosaic=1.0, scale=0.3,
        hsv_h=0.01, hsv_s=0.3, hsv_v=0.3,
    )


if __name__ == "__main__":
    main()
