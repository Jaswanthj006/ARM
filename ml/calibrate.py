"""Confidence calibration: turn YOLO's raw score into a real-world accuracy %.

Runs the detector the way the phone does (photo scaled to 1920 px, 640 px tiles, stride 480) on the
held-out boards 10-12 (never seen in training) and records, for every detection, whether it was
correct (same class, IoU >= 0.5 with a labelled defect).

Per defect type it fits a monotonic curve  raw score -> fraction correct  (isotonic regression,
pooled with all types when a type has few detections). Reliability is checked honestly with a
2-fold split by board: fit on half the boards, measure on the other half.

Writes  web/js/calibration.json  (used by the website) and  ml/results/calibration.json  (report).
Usage:  python3 ml/calibrate.py
"""
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from eval_boards import SRC, detect_board, iou

ROOT = Path(__file__).resolve().parent
MAX_SIDE, CONF = 1920, 0.20
GRID = [round(x, 2) for x in np.arange(0.20, 1.0001, 0.05)]


def collect(model, device, names, stems):
    out = []   # (board, class name, score, correct)
    for k, stem in enumerate(stems):
        img = cv2.imread(str(next(SRC.glob(f"images/**/{stem}.*"))))
        if max(img.shape[:2]) > MAX_SIDE:
            f = MAX_SIDE / max(img.shape[:2])
            img = cv2.resize(img, (int(img.shape[1] * f), int(img.shape[0] * f)), interpolation=cv2.INTER_AREA)
        H, W = img.shape[:2]
        gt = []
        for line in next(SRC.glob(f"labels/**/{stem}.txt")).read_text().split("\n"):
            if line.strip():
                c, cx, cy, bw, bh = map(float, line.split())
                gt.append((int(c), (cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W, (cy + bh / 2) * H))
        used = set()
        for c, s, *bx in sorted(detect_board(model, img, CONF, device), key=lambda d: -d[1]):
            best, bi = 0, None
            for j, g in enumerate(gt):
                if j not in used and g[0] == c and iou(bx, g[1:]) > best:
                    best, bi = iou(bx, g[1:]), j
            ok = best >= 0.5
            if ok:
                used.add(bi)
            out.append((stem, names[c], s, int(ok)))
        if k % 20 == 0:
            print(f"  {k}/{len(stems)} boards", flush=True)
    return out


def isotonic(x, y):
    """Pool-adjacent-violators: non-decreasing fit of y on sorted x."""
    o = np.argsort(x); x, y = np.asarray(x)[o], np.asarray(y, float)[o]
    blocks = [[v, 1.0] for v in y]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0]:
            m = (blocks[i][0] * blocks[i][1] + blocks[i + 1][0] * blocks[i + 1][1]) / (blocks[i][1] + blocks[i + 1][1])
            blocks[i] = [m, blocks[i][1] + blocks[i + 1][1]]; del blocks[i + 1]; i = max(i - 1, 0)
        else:
            i += 1
    fit = np.concatenate([[m] * int(n) for m, n in blocks])
    return x, fit


def curve(samples):
    """Calibration curve on GRID: accuracy for each raw-score grid point."""
    s = np.array([d[2] for d in samples]); y = np.array([d[3] for d in samples])
    if len(s) < 5:
        return None
    x, f = isotonic(s, y)
    return [float(np.clip(np.interp(g, x, f), 0.0, 1.0)) for g in GRID]


def fit(samples, names):
    pooled = curve(samples)
    curves = {}
    for n in names:
        sub = [d for d in samples if d[1] == n]
        c = curve(sub)
        w = min(1.0, len(sub) / 60)                      # few detections of a type -> lean on the pooled curve
        curves[n] = [round(w * a + (1 - w) * b, 3) for a, b in zip(c, pooled)] if c else [round(b, 3) for b in pooled]
    return curves


def apply(curves, name, s):
    return float(np.interp(s, GRID, curves[name]))


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = YOLO(str(ROOT / "runs/pcb_yolo11n/weights/best.pt"))
    names = [l.strip() for l in (SRC / "classes.txt").read_text().splitlines() if l.strip()]
    stems = [Path(l.strip()).stem for l in (SRC / "ImageSets/val_board.txt").read_text().splitlines() if l.strip()]
    print(f"collecting detections on {len(stems)} held-out boards …")
    S = collect(model, device, names, stems)

    # Honest reliability: 2-fold split by board (calibrate on one half, measure on the other).
    boards = sorted({d[0] for d in S}); half = set(boards[::2])
    A = [d for d in S if d[0] in half]; B = [d for d in S if d[0] not in half]
    bins = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
    rel, err_raw, err_cal, n = [], 0.0, 0.0, 0
    for tr, te in ((A, B), (B, A)):
        cv = fit(tr, names)
        for d in te:
            p = apply(cv, d[1], d[2]); err_cal += abs(p - d[3]); err_raw += abs(d[2] - d[3]); n += 1
        for lo, hi in bins:
            sel = [d for d in te if lo <= apply(cv, d[1], d[2]) < hi]
            if sel:
                rel.append((lo, hi, len(sel), float(np.mean([apply(cv, d[1], d[2]) for d in sel])), float(np.mean([d[3] for d in sel]))))
    table = []
    for lo, hi in bins:
        r = [x for x in rel if x[0] == lo]
        if r:
            cnt = sum(x[2] for x in r)
            table.append(dict(stated=f"{int(lo*100)}-{min(100, int(hi*100))}%", detections=cnt,
                              avg_stated=round(sum(x[2] * x[3] for x in r) / cnt, 3), actually_correct=round(sum(x[2] * x[4] for x in r) / cnt, 3)))
    # Expected calibration error (lower = stated % closer to truth)
    def ece(pred):
        tot, e = 0, 0.0
        for lo, hi in bins:
            sel = [(p, y) for p, y in pred if lo <= p < hi]
            if sel:
                e += len(sel) * abs(np.mean([p for p, _ in sel]) - np.mean([y for _, y in sel])); tot += len(sel)
        return e / max(tot, 1)
    preds_raw = [(d[2], d[3]) for d in S]
    preds_cal = []
    for tr, te in ((A, B), (B, A)):
        cv = fit(tr, names); preds_cal += [(apply(cv, d[1], d[2]), d[3]) for d in te]

    final = fit(S, names)                                   # deployed curve uses all boards
    per_type = {n: dict(detections=sum(d[1] == n for d in S), correct=sum(d[3] for d in S if d[1] == n)) for n in names}
    report = dict(boards=len(stems), detections=len(S), min_score=CONF, max_side=MAX_SIDE,
                  calibration_error_raw_score=round(ece(preds_raw), 3), calibration_error_calibrated=round(ece(preds_cal), 3),
                  reliability_held_out_half=table, per_type=per_type,
                  note="Calibrated on PKU held-out boards 10-12 (dataset photos). Real phone-camera photos may differ; re-calibrate on real captures.")
    web = dict(grid=GRID, curves={**final, "open": final.get("open_circuit")}, report=report)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/calibration.json").write_text(json.dumps(dict(report=report, curves=final, grid=GRID), indent=1))
    (ROOT.parent / "web/js/calibration.json").write_text(json.dumps(web))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
