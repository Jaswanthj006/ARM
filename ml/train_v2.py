"""BoardCheck detector v2: YOLO11n on PKU + DeepPCB + DsPCBSD+ with phone-camera augmentation.

What is new compared with ml/train.py
  * data: ml/data/merged (built by ml/data_prep/build_merged.py)
  * missing_hole tiles (PKU only, the rarest class) are listed 3x in the training list
  * real clean-board crops from the RED board video are added as background images
    (the GREEN board is kept out of training: it is the clean-board test)
  * PhoneLook (ml/augment.py): blur, motion blur, low resolution, noise, JPEG, glare, shadow,
    white balance, grayscale, and a new board colour on ~24 % of tiles (red / blue / black masks),
    small rotation, shear and perspective for tilted handheld photos
  * starts from the current best model (ml/runs/pcb_yolo11n/weights/best.pt)
  * validates on PKU boards 10-12 only during training (full 3-source test is Phase 3)

Usage:
  python3 -I ml/train_v2.py [--epochs 25] [--fraction 1.0] [--name pcb_yolo11n_v2] [--resume]
"""
import argparse
import sys
from pathlib import Path

ML = Path(__file__).resolve().parent
sys.path.insert(0, str(ML))

import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402

import augment  # noqa: E402

M = ML / "data" / "merged"
NAMES = ["missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper", "missing_pad"]


def prepare(fraction: float, hardneg_repeat: int = 0):
    """Write the training list and data yaml. Returns the yaml path."""
    neg_img, neg_lab = M / "images" / "neg_real", M / "labels" / "neg_real"
    neg_img.mkdir(exist_ok=True); neg_lab.mkdir(exist_ok=True)
    for src in sorted(p for p in (ML / "data" / "real_video").glob("*/v_red_*.jpg") if not p.name.startswith(".")):
        dst = neg_img / src.name
        if not dst.exists():
            dst.symlink_to(src)
        (neg_lab / f"{src.stem}.txt").write_text("")
    train = sorted(p for p in (M / "images" / "train").glob("*.jpg") if not p.name.startswith("."))
    if fraction < 1.0:
        train = train[:: max(1, round(1 / fraction))]
    extra = []
    for p in train:
        if p.name.startswith("pku_"):
            lab = (M / "labels" / "train" / f"{p.stem}.txt").read_text()
            if any(l.split()[0] == "0" for l in lab.splitlines() if l.strip()):
                extra += [p, p]
    negs = sorted(p for p in neg_img.glob("*.jpg") if not p.name.startswith("."))
    hard = []
    if hardneg_repeat and (M / "images" / "hardneg").is_dir():
        # false-alarm tiles mined from TRAINING data only (ml/data_prep/mine_hard_negatives.py)
        hard = sorted(p for p in (M / "images" / "hardneg").glob("*.jpg") if not p.name.startswith(".")) * hardneg_repeat
        lst_file = M / "hardneg_existing.txt"
        if lst_file.exists():
            hard += [M / l.strip() for l in lst_file.read_text().splitlines() if l.strip()]
    lst = train + extra + negs + hard
    (M / "train_v2.txt").write_text("\n".join(str(p) for p in lst) + "\n")
    yaml = M / "data_v2.yaml"
    yaml.write_text(f"path: {M}\ntrain: train_v2.txt\nval:\n  - images/val_pku\n  - images/val_pcbdefect\n  - images/val_defectdet\nnames:\n"
                    + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)))
    print(f"train list: {len(train)} images + {len(extra)} extra missing_hole + {len(negs)} real clean crops "
          f"+ {len(hard)} hard-negative entries = {len(lst)}")
    return yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--fraction", type=float, default=1.0, help="use a subset of the training images (smoke test)")
    ap.add_argument("--name", default="pcb_yolo11n_v2")
    ap.add_argument("--model", default=str(ML / "runs" / "pcb_yolo11n" / "weights" / "best.pt"))
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--aug", choices=("full", "plain", "nophone"), default="full",
                    help="ablation: plain = old train.py augmentation; nophone = full minus PhoneLook")
    ap.add_argument("--hardneg", type=int, default=0, help="repeat mined false-alarm tiles this many times (0 = off)")
    ap.add_argument("--lr0", type=float, default=0.002)
    ap.add_argument("--close-mosaic", type=int, default=5)
    a = ap.parse_args()
    device = 0 if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    if a.aug == "full":
        augment.install(p=0.8)
    geo = dict(degrees=5.0, shear=2.0, perspective=0.0005, scale=0.5, translate=0.1)
    col = dict(hsv_h=0.015, hsv_s=0.5, hsv_v=0.4)   # other board colours come from PhoneLook (24 % of tiles)
    if a.aug == "plain":
        geo = dict(degrees=0.0, shear=0.0, perspective=0.0, scale=0.3, translate=0.1)
        col = dict(hsv_h=0.01, hsv_s=0.3, hsv_v=0.3)
    if a.resume:
        YOLO(str(ML / "runs" / a.name / "weights" / "last.pt")).train(resume=True, device=device)
        return
    yaml = prepare(a.fraction, a.hardneg)
    YOLO(a.model).train(
        data=str(yaml), imgsz=640, epochs=a.epochs, batch=a.batch, device=device, workers=a.workers,
        project=str(ML / "runs"), name=a.name, exist_ok=True, seed=0,
        patience=10, cos_lr=True, close_mosaic=a.close_mosaic, save_period=1,
        # fine-tuning an already trained model: gentle learning rate, no bias warm-up jump
        optimizer="SGD", lr0=a.lr0, lrf=0.1, warmup_epochs=1, warmup_bias_lr=0.0,
        # geometry: flips + small tilt / rotation for handheld photos; scale for distance changes
        fliplr=0.5, flipud=0.5, **geo,
        # colour: any hue (green / red / blue / black boards), stronger saturation and brightness
        **col,
        mosaic=1.0, mixup=0.0,
    )


if __name__ == "__main__":
    main()
