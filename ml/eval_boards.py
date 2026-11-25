"""Evaluate the detector the way it runs on the phone: whole board, tiled.

For every validation board (boards 10-12, never seen in training):
  1. cut into 640 px tiles with 160 px overlap,
  2. run the detector on each tile,
  3. merge boxes back to board coordinates with class-wise NMS,
  4. match to ground truth at IoU >= 0.5.

Reports per-class precision / recall, missed defects and false alarms, plus
board-level accuracy (does the board get flagged when it has defects).

Usage:  python3 ml/eval_boards.py --weights ml/runs/pcb_yolo11n/weights/best.pt [--conf 0.25]
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "data" / "pku"
import os
TILE, STRIDE = 640, int(os.environ.get("STRIDE", 480))


def starts(size):
    s = list(range(0, max(size - TILE, 0) + 1, STRIDE))
    if s[-1] + TILE < size:
        s.append(size - TILE)
    return s


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def detect_board(model, img, conf, device):
    H, W = img.shape[:2]
    tiles, offs = [], []
    for y0 in starts(H):
        for x0 in starts(W):
            tiles.append(img[y0:y0 + TILE, x0:x0 + TILE]); offs.append((x0, y0))
    res = model.predict(tiles, imgsz=TILE, conf=conf, device=device, verbose=False, batch=len(tiles))
    boxes, scores, cls = [], [], []
    for r, (x0, y0) in zip(res, offs):
        if r.boxes is None or not len(r.boxes):
            continue
        b = r.boxes.xyxy.cpu().numpy(); b[:, [0, 2]] += x0; b[:, [1, 3]] += y0
        boxes.append(b); scores.append(r.boxes.conf.cpu().numpy()); cls.append(r.boxes.cls.cpu().numpy())
    if not boxes:
        return []
    b, s, c = (np.concatenate(x) for x in (boxes, scores, cls))
    keep = torchvision.ops.batched_nms(torch.tensor(b), torch.tensor(s), torch.tensor(c).long(), 0.4).numpy()
    return [(int(c[i]), float(s[i]), *b[i].tolist()) for i in keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "runs/pcb_yolo11n/weights/best.pt"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default=str(ROOT / "results" / "eval_boards.json"))
    ap.add_argument("--max-side", type=int, default=0, help="downscale the longest side first, like the phone (PHOTO_MAX)")
    ap.add_argument("--class-conf", default="", help='per-class thresholds, e.g. {"short":0.25,"spur":0.4}')
    a = ap.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = YOLO(a.weights)
    class_conf = json.loads(a.class_conf) if a.class_conf else {}
    names = [l.strip() for l in (SRC / "classes.txt").read_text().splitlines() if l.strip()]
    stems = [Path(l.strip()).stem for l in (SRC / "ImageSets/val_board.txt").read_text().splitlines() if l.strip()]

    tp = {n: 0 for n in names}; fp = dict(tp); fn = dict(tp)
    times, flagged = [], 0
    for stem in stems:
        img = cv2.imread(str(next(SRC.glob(f"images/**/{stem}.*"))))
        if a.max_side and max(img.shape[:2]) > a.max_side:
            k = a.max_side / max(img.shape[:2])
            img = cv2.resize(img, (int(img.shape[1] * k), int(img.shape[0] * k)), interpolation=cv2.INTER_AREA)
        H, W = img.shape[:2]
        gt = []
        for line in next(SRC.glob(f"labels/**/{stem}.txt")).read_text().split("\n"):
            if line.strip():
                c, cx, cy, bw, bh = map(float, line.split())
                gt.append((int(c), (cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W, (cy + bh / 2) * H))
        t = time.perf_counter(); det = detect_board(model, img, a.conf, device); times.append(time.perf_counter() - t)
        if class_conf:
            det = [d for d in det if d[1] >= class_conf.get(names[d[0]], a.conf)]
        flagged += bool(det)
        used = set()
        for c, s, *bx in sorted(det, key=lambda d: -d[1]):
            best, bi = 0, None
            for j, g in enumerate(gt):
                if j in used or g[0] != c:
                    continue
                v = iou(bx, g[1:])
                if v > best:
                    best, bi = v, j
            if best >= 0.5:
                used.add(bi); tp[names[c]] += 1
            else:
                fp[names[c]] += 1
        for j, g in enumerate(gt):
            if j not in used:
                fn[names[g[0]]] += 1

    rows = {}
    for n in names:
        p = tp[n] / (tp[n] + fp[n]) if tp[n] + fp[n] else 0.0
        r = tp[n] / (tp[n] + fn[n]) if tp[n] + fn[n] else 0.0
        rows[n] = dict(precision=round(p, 3), recall=round(r, 3), found=tp[n], missed=fn[n], false_alarms=fp[n])
    T, F, M = sum(tp.values()), sum(fp.values()), sum(fn.values())
    summary = dict(
        boards=len(stems), max_side=a.max_side or "full", conf=a.conf, class_conf=class_conf, defects=T + M,
        precision=round(T / (T + F), 3) if T + F else 0, recall=round(T / (T + M), 3) if T + M else 0,
        missed=M, false_alarms=F, boards_flagged=f"{flagged}/{len(stems)}",
        mac_ms_per_board=round(1000 * float(np.median(times)), 1), device=device,
        note="Every validation board has defects, so board-level false rejects cannot be measured on this set.",
    )
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(summary=summary, per_class=rows), indent=2))
    print(json.dumps(summary, indent=2))
    print(f"{'class':18}{'prec':>7}{'recall':>8}{'found':>7}{'missed':>8}{'false':>7}")
    for n, r in rows.items():
        print(f"{n:18}{r['precision']:7.3f}{r['recall']:8.3f}{r['found']:7}{r['missed']:8}{r['false_alarms']:7}")


if __name__ == "__main__":
    main()
