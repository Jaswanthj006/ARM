"""Phase 3: compare detector versions on every test set we have.

Models (name=path) are run once per image at a low score (0.05); everything else is computed from
the cached detections, so thresholds can be tuned without re-running the models.

Test sets
  pku_boards  PKU boards 10-12, whole boards tiled like the app (152 boards, 768 defects)
  deeppcb     DeepPCB test split, 2000 tiles   (unseen by the old model's training AND by v2's)
  dspcbsd     DsPCBSD+ test split, 993 tiles
  green       real clean board, handheld phone video (37 crops; never used in training)
  phone       real clean board frames from the phone app (82 frames, 480x640)

Matching: same class and IoU >= 0.5 (also reported at IoU >= 0.3, because PKU boxes are drawn
loosely and DsPCBSD+/DeepPCB boxes tightly, so a correct but tighter box can fall below 0.5).

Threshold tuning: PKU boards are split by board name into two halves (alternate images).
Per-class thresholds are tuned on half A (max F1; for the critical defects open_circuit and short
the threshold is lowered until recall >= 0.95 if possible) and reported on half B only.

Usage: python3 -I ml/eval_v2.py old=ml/runs/pcb_yolo11n/weights/best.pt new=... [more=...]
Output: ml/results/phase3_eval.json and a printed summary.
"""
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from ultralytics import YOLO

ML = Path(__file__).resolve().parent
PKU = ML / "data" / "pku"
M = ML / "data" / "merged"
NAMES = ["missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper", "missing_pad"]
APP_THR = {"missing_hole": 0.40, "mouse_bite": 0.40, "open_circuit": 0.40, "short": 0.25, "spur": 0.40, "spurious_copper": 0.40,
           "missing_pad": 0.40}
CRITICAL = ("open_circuit", "short")
TILE, STRIDE, LOW = 640, 480, 0.05
DEV = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
CACHE = ML / "results" / "phase3_cache"


def starts(size):
    s = list(range(0, max(size - TILE, 0) + 1, STRIDE))
    if s[-1] + TILE < size:
        s.append(size - TILE)
    return s


def detect(model, img):
    H, W = img.shape[:2]
    tiles, offs = [], []
    for y0 in starts(H):
        for x0 in starts(W):
            tiles.append(img[y0:y0 + TILE, x0:x0 + TILE]); offs.append((x0, y0))
    b_all, s_all, c_all = [], [], []
    for i in range(0, len(tiles), 16):
        for r, (x0, y0) in zip(model.predict(tiles[i:i + 16], imgsz=TILE, conf=LOW, device=DEV, verbose=False), offs[i:i + 16]):
            if r.boxes is None or not len(r.boxes):
                continue
            b = r.boxes.xyxy.cpu().numpy(); b[:, [0, 2]] += x0; b[:, [1, 3]] += y0
            b_all.append(b); s_all.append(r.boxes.conf.cpu().numpy()); c_all.append(r.boxes.cls.cpu().numpy())
    if not b_all:
        return []
    b, s, c = (np.concatenate(x) for x in (b_all, s_all, c_all))
    keep = torchvision.ops.batched_nms(torch.tensor(b), torch.tensor(s), torch.tensor(c).long(), 0.4).numpy()
    return [(int(c[i]), float(s[i]), *map(float, b[i])) for i in keep]


def yolo_gt(lab, W, H):
    out = []
    if lab.exists():
        for l in lab.read_text().split("\n"):
            if l.strip():
                c, cx, cy, bw, bh = map(float, l.split()[:5])
                out.append((int(c), (cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W, (cy + bh / 2) * H))
    return out


def test_sets():
    sets = {}
    stems = [Path(l.strip()).stem for l in (PKU / "ImageSets/val_board.txt").read_text().splitlines() if l.strip()]
    sets["pku_boards"] = [(s, next(PKU.glob(f"images/**/{s}.*")), next(PKU.glob(f"labels/**/{s}.txt"))) for s in stems]
    for name, folder in (("deeppcb", "val_deeppcb"), ("dspcbsd", "val_dspcbsd")):
        sets[name] = [(p.stem, p, M / "labels" / folder / f"{p.stem}.txt")
                      for p in sorted((M / "images" / folder).glob("*.jpg")) if not p.name.startswith(".")]
    for name in ("pcbdefect", "defectdet"):   # whole real boards from designs never used in training
        root = ML / "data" / "real_test" / name
        sets["real_" + name] = [(p.stem, p, root / "labels" / f"{p.stem}.txt")
                                for p in sorted((root / "images").glob("*.jpg")) if not p.name.startswith(".")]
    sets["green"] = [(p.stem, p, None) for p in sorted((ML / "data" / "real_video").glob("*/v_green_*.jpg"))]
    sets["phone"] = [(p.stem, p, None) for p in sorted((ML / "data" / "pcb_real" / "phone").glob("*.jpg"))]
    return sets


def run_model(name, path, sets):
    f = CACHE / f"{name}.pkl"
    out = pickle.loads(f.read_bytes()) if f.exists() else {}
    missing = [s for s in sets if s not in out]
    if not missing:
        return out
    model = YOLO(path)
    for sname in missing:          # only run the model on test sets not cached yet
        items = sets[sname]
        out[sname] = {}
        for stem, img_p, lab_p in items:
            img = cv2.imread(str(img_p))
            H, W = img.shape[:2]
            out[sname][stem] = dict(det=detect(model, img), gt=yolo_gt(lab_p, W, H) if lab_p else [])
        print(f"  {name}: {sname} done ({len(items)} images)", flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    f.write_bytes(pickle.dumps(out))
    return out


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def score(records, thr, iou_min=0.5, keys=None):
    tp = {n: 0 for n in NAMES}; fp = dict(tp); fn = dict(tp)
    flagged = 0
    for k, rec in records.items():
        if keys is not None and k not in keys:
            continue
        det = [d for d in rec["det"] if d[1] >= thr[NAMES[d[0]]]]
        flagged += bool(det)
        used = set()
        for c, s, *bx in sorted(det, key=lambda d: -d[1]):
            best, bi = 0.0, None
            for j, g in enumerate(rec["gt"]):
                if j in used or g[0] != c:
                    continue
                v = iou(bx, g[1:])
                if v > best:
                    best, bi = v, j
            if best >= iou_min:
                used.add(bi); tp[NAMES[c]] += 1
            else:
                fp[NAMES[c]] += 1
        for j, g in enumerate(rec["gt"]):
            if j not in used:
                fn[NAMES[g[0]]] += 1
    T, F, Mi = sum(tp.values()), sum(fp.values()), sum(fn.values())
    per = {n: dict(P=round(tp[n] / (tp[n] + fp[n]), 3) if tp[n] + fp[n] else None,
                   R=round(tp[n] / (tp[n] + fn[n]), 3) if tp[n] + fn[n] else None,
                   found=tp[n], missed=fn[n], false=fp[n]) for n in NAMES}
    n_img = len(records) if keys is None else len(keys)
    return dict(P=round(T / (T + F), 3) if T + F else None, R=round(T / (T + Mi), 3) if T + Mi else None,
                found=T, missed=Mi, false=F, images=n_img, images_flagged=flagged, per_class=per)


def tune(records, keys):
    thr = dict(APP_THR)
    grid = [round(x, 2) for x in np.arange(0.10, 0.81, 0.05)]
    for n in NAMES:
        best = None
        for t in grid:
            th = dict(thr); th[n] = t
            r = score(records, th, keys=keys)["per_class"][n]
            p, rc = r["P"] or 0.0, r["R"] or 0.0
            f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
            ok = rc >= 0.95 if n in CRITICAL else True
            key = (ok, f1)
            if best is None or key > best[0]:
                best = (key, t)
        thr[n] = best[1]
    return thr


def main():
    models = dict(a.split("=", 1) for a in sys.argv[1:])
    sets = test_sets()
    print("test sets:", {k: len(v) for k, v in sets.items()})
    stems = sorted(sets["pku_boards_keys"] if False else [s for s, _, _ in sets["pku_boards"]])
    half_a, half_b = set(stems[0::2]), set(stems[1::2])
    report = {"note": __doc__.split("Usage")[0].strip(), "models": models, "app_thresholds": APP_THR, "results": {}}
    for name, path in models.items():
        print(f"running {name} ...", flush=True)
        rec = run_model(name, path, sets)
        tuned = tune(rec["pku_boards"], half_a)
        res = {"tuned_thresholds_on_pku_half_A": tuned}
        for thr_name, thr in (("app", APP_THR), ("tuned", tuned)):
            r = {}
            r["pku_boards_all"] = score(rec["pku_boards"], thr)
            r["pku_boards_all_iou0.3"] = score(rec["pku_boards"], thr, iou_min=0.3)
            r["pku_boards_half_B"] = score(rec["pku_boards"], thr, keys=half_b)
            for s in ("deeppcb", "dspcbsd", "real_pcbdefect", "real_defectdet"):
                r[s] = score(rec[s], thr)
                r[s + "_iou0.3"] = score(rec[s], thr, iou_min=0.3)
            for s in ("green", "phone"):
                x = score(rec[s], thr)
                r[s + "_clean"] = dict(images=x["images"], images_with_false_alarm=x["images_flagged"], false_boxes=x["false"],
                                       by_class={n: v["false"] for n, v in x["per_class"].items() if v["false"]})
            res[thr_name] = r
        report["results"][name] = res
    (ML / "results" / "phase3_eval.json").write_text(json.dumps(report, indent=1))
    # printed summary
    print("\n=== summary (P / R; clean = images with any false alarm) ===")
    for name, res in report["results"].items():
        for thr_name in ("app", "tuned"):
            r = res[thr_name]
            f = lambda k: f"{r[k]['P']}/{r[k]['R']}"
            print(f"{name:8} {thr_name:6} PKU {f('pku_boards_all')} (iou.3 {f('pku_boards_all_iou0.3')}) "
                  f"halfB {f('pku_boards_half_B')} | DeepPCB {f('deeppcb')} (iou.3 {f('deeppcb_iou0.3')}) "
                  f"| DsPCBSD {f('dspcbsd')} (iou.3 {f('dspcbsd_iou0.3')}) "
                  f"| realPCBD {f('real_pcbdefect')} (iou.3 {f('real_pcbdefect_iou0.3')}) "
                  f"| realDDET {f('real_defectdet')} (iou.3 {f('real_defectdet_iou0.3')}) "
                  f"| green {r['green_clean']['images_with_false_alarm']}/{r['green_clean']['images']} "
                  f"| phone {r['phone_clean']['images_with_false_alarm']}/{r['phone_clean']['images']}")
        print(f"{name:8} tuned thresholds: {res['tuned_thresholds_on_pku_half_A']}")


if __name__ == "__main__":
    main()
