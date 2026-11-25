"""False-alarm mining for the hard-negative training round (training data only, never the test boards).

1. PKU training boards (01, 04-09): every 640 px tile (stride 480), INCLUDING the defect-free tiles that
   prepare_tiles.py mostly dropped. A detection is a false alarm if no ground-truth box of the same class
   overlaps it at IoU >= 0.3 and it does not touch any ground-truth box (IoU > 0.1) of another class.
   Tiles with a false alarm are written to merged/{images,labels}/hardneg with their correct labels
   (empty label = clean tile), so the model is shown "this is normal board, not a defect".
2. DeepPCB / DsPCBSD+ training tiles already in merged/train: tiles with a false alarm are listed in
   merged/hardneg_existing.txt so the training list can repeat them.

Usage: python3 -I ml/data_prep/mine_hard_negatives.py --weights ml/runs/pcb_yolo11n_v2_aws/weights/best.pt
"""
import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

ML = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ML))
from eval_v2 import iou, yolo_gt  # noqa: E402

PKU = ML / "data" / "pku"
M = ML / "data" / "merged"
NAMES = ["missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"]
TILE, STRIDE, MIN_KEEP = 640, 480, 0.6
DEV = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


def starts(size):
    s = list(range(0, max(size - TILE, 0) + 1, STRIDE))
    if s[-1] + TILE < size:
        s.append(size - TILE)
    return s


def false_alarms(dets, gts):
    out = []
    for c, s, *b in dets:
        same = max([iou(b, g[1:]) for g in gts if g[0] == c] or [0.0])
        anyg = max([iou(b, g[1:]) for g in gts] or [0.0])
        if same < 0.3 and anyg <= 0.1:
            out.append((c, s))
    return out


def predict(model, imgs, conf):
    res = model.predict(imgs, imgsz=TILE, conf=conf, device=DEV, verbose=False)
    return [[(int(k), float(s), *map(float, b)) for b, s, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())] for r in res]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--conf", type=float, default=0.35)
    a = ap.parse_args()
    model = YOLO(a.weights)
    out_i, out_l = M / "images" / "hardneg", M / "labels" / "hardneg"
    for d in (out_i, out_l):
        shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
    stats = Counter(); by_class = Counter()

    # 1. PKU training boards, all tiles
    stems = [Path(l.strip()).stem for l in (PKU / "ImageSets" / "train_board.txt").read_text().splitlines() if l.strip()]
    for n, stem in enumerate(stems):
        img = cv2.imread(str(next(PKU.glob(f"images/**/{stem}.*"))))
        H, W = img.shape[:2]
        gts = yolo_gt(next(PKU.glob(f"labels/**/{stem}.txt")), W, H)
        offs = [(x0, y0) for y0 in starts(H) for x0 in starts(W)]
        tiles = [img[y0:y0 + TILE, x0:x0 + TILE] for x0, y0 in offs]
        for i in range(0, len(tiles), 16):
            for (x0, y0), tile, dets in zip(offs[i:i + 16], tiles[i:i + 16], predict(model, tiles[i:i + 16], a.conf)):
                tg = []
                for c, x1, y1, x2, y2 in gts:   # ground truth clipped to this tile (same rule as prepare_tiles.py)
                    ia, ib, ic, idd = max(x1, x0), max(y1, y0), min(x2, x0 + TILE), min(y2, y0 + TILE)
                    if ic > ia and idd > ib:
                        tg.append((c, ia - x0, ib - y0, ic - x0, idd - y0, (ic - ia) * (idd - ib) >= MIN_KEEP * (x2 - x1) * (y2 - y1)))
                fa = false_alarms(dets, [g[:5] for g in tg])
                stats["pku_tiles"] += 1
                if not fa:
                    continue
                lines = [f"{c} {(a_ + c_) / 2 / TILE:.6f} {(b_ + d_) / 2 / TILE:.6f} {(c_ - a_) / TILE:.6f} {(d_ - b_) / TILE:.6f}"
                         for c, a_, b_, c_, d_, keep in tg if keep]
                name = f"hn_pku_{stem}_{x0}_{y0}"
                cv2.imwrite(str(out_i / f"{name}.jpg"), tile, [cv2.IMWRITE_JPEG_QUALITY, 95])
                (out_l / f"{name}.txt").write_text("\n".join(lines))
                stats["pku_hardneg_tiles"] += 1
                stats["pku_hardneg_clean_tiles"] += not lines
                for c, _ in fa:
                    by_class[NAMES[c]] += 1
        if n % 100 == 0:
            print(f"  PKU {n}/{len(stems)} boards, {stats['pku_hardneg_tiles']} hard tiles so far", flush=True)

    # 2. DeepPCB / DsPCBSD+ training tiles
    existing = []
    imgs = sorted(p for p in (M / "images" / "train").glob("*.jpg") if not p.name.startswith(".") and not p.name.startswith("pku_"))
    for i in range(0, len(imgs), 32):
        batch = imgs[i:i + 32]
        arrs = [cv2.imread(str(p)) for p in batch]
        for p, arr, dets in zip(batch, arrs, predict(model, arrs, a.conf)):
            gts = yolo_gt(M / "labels" / "train" / f"{p.stem}.txt", arr.shape[1], arr.shape[0])
            fa = false_alarms(dets, gts)
            stats["other_tiles"] += 1
            if fa:
                existing.append(str(p.relative_to(M)))
                for c, _ in fa:
                    by_class[NAMES[c] + "_other"] += 1
    (M / "hardneg_existing.txt").write_text("\n".join(existing) + "\n")
    stats["other_hardneg_tiles"] = len(existing)
    report = dict(weights=a.weights, conf=a.conf, stats=dict(stats), false_alarms_by_class=dict(by_class))
    (ML / "results" / "hard_negatives.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
