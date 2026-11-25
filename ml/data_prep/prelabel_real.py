"""Pre-label real phone photos with the current detector, ready to correct in CVAT / Label Studio / Roboflow.

Photo names must follow   <board>_<condition>_<nn>.jpg   e.g. b07_dim_03.jpg, b12_clean_01.jpg
  board      b01 ... bNN (one id per physical board; boards marked 'clean' have no defects)
  condition  normal | dim | bright | tilt | hand | clean

Runs the same 640 px tiling as the app (stride 480) on the full photo, writes YOLO labels next to a
copy of each photo, and splits by BOARD (never by photo): boards listed in --test go to the test set.

Usage:
  python3 -I ml/data_prep/prelabel_real.py <photo_dir> --test b03,b08,b11 [--weights ml/runs/pcb_yolo11n/weights/best.pt]
Output: ml/data/real/{images,labels}/{train,test}/ and ml/data/real/manifest.json
"""
import argparse
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

ML = Path(__file__).resolve().parent.parent
OUT = ML / "data" / "real"
TILE, STRIDE = 640, 480
THRESH = {0: 0.40, 1: 0.40, 2: 0.40, 3: 0.25, 4: 0.40, 5: 0.40}   # same as the app
NAME_RE = re.compile(r"^(b\d+)_([a-z]+)_(\d+)$")


def starts(size):
    s = list(range(0, max(size - TILE, 0) + 1, STRIDE))
    if s[-1] + TILE < size:
        s.append(size - TILE)
    return s


def detect(model, img, device):
    H, W = img.shape[:2]
    boxes, confs, cls = [], [], []
    for y0 in starts(H):
        for x0 in starts(W):
            r = model.predict(img[y0:y0 + TILE, x0:x0 + TILE], imgsz=TILE, conf=0.25, device=device, verbose=False)[0]
            for b, c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
                boxes.append([b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0]); confs.append(c); cls.append(int(k))
    if not boxes:
        return []
    import torchvision
    b, c, k = torch.tensor(boxes), torch.tensor(confs), torch.tensor(cls)
    keep = torchvision.ops.batched_nms(b, c, k, 0.4)
    return [(int(k[i]), *b[i].tolist()) for i in keep if c[i] >= THRESH[int(k[i])]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("photos", type=Path)
    ap.add_argument("--test", required=True, help="comma-separated board ids held out for testing")
    ap.add_argument("--weights", default=str(ML / "runs" / "pcb_yolo11n" / "weights" / "best.pt"))
    a = ap.parse_args()
    test_boards = {t.strip() for t in a.test.split(",") if t.strip()}
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = YOLO(a.weights)
    for sp in ("train", "test"):
        (OUT / "images" / sp).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / sp).mkdir(parents=True, exist_ok=True)
    manifest, bad, per = [], [], defaultdict(Counter)
    for p in sorted(a.photos.iterdir()):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        m = NAME_RE.match(p.stem.lower())
        if not m:
            bad.append(p.name)
            continue
        board, cond, _ = m.groups()
        sp = "test" if board in test_boards else "train"
        img = cv2.imread(str(p))
        if img is None:
            bad.append(p.name)
            continue
        H, W = img.shape[:2]
        dets = detect(model, img, device)
        lines = [f"{c} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}"
                 for c, x1, y1, x2, y2 in dets]
        shutil.copy2(p, OUT / "images" / sp / f"{p.stem.lower()}.jpg")
        (OUT / "labels" / sp / f"{p.stem.lower()}.txt").write_text("\n".join(lines))
        manifest.append(dict(file=p.name, board=board, condition=cond, split=sp, prelabels=len(lines), size=[W, H]))
        per[sp][cond] += 1
    (OUT / "manifest.json").write_text(json.dumps(dict(test_boards=sorted(test_boards), photos=manifest,
                                                       per_split={k: dict(v) for k, v in per.items()},
                                                       skipped_bad_name=bad), indent=1))
    print(f"{len(manifest)} photos pre-labelled -> {OUT}  per split: { {k: dict(v) for k, v in per.items()} }")
    if bad:
        print("skipped (name must be <board>_<condition>_<nn>.jpg):", bad[:10])
    print("Next: correct the labels in your labelling tool (YOLO format), then re-save into the same folder.")


if __name__ == "__main__":
    main()
