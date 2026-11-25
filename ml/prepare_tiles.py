"""Cut PKU-Market-PCB images into overlapping 640 px tiles for YOLO training.

PCB defects are a few dozen pixels on ~3000 x 1600 images. Training and running
the detector on tiles keeps them visible at 640 px input. The phone app uses the
same tile size and overlap at inference time.

Uses the board-disjoint split (train boards 01,04-09 / val boards 10-12) so the
validation numbers are not inflated by boards the model has already seen.

Usage:  python3 ml/prepare_tiles.py
"""
import random
import shutil
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "data" / "pku"
OUT = ROOT / "data" / "tiles"
TILE, STRIDE = 640, 480          # 160 px overlap
MIN_KEEP = 0.6                   # keep a box if >= 60 % of it lies inside the tile
NEG_KEEP = {"train": 0.12, "val": 0.12}   # share of defect-free tiles to keep
random.seed(0)


def find_label(stem: str) -> Path:
    for cand in (SRC / "labels" / f"{stem}.txt", *SRC.glob(f"labels/**/{stem}.txt")):
        if cand.exists():
            return cand
    raise FileNotFoundError(stem)


def find_image(stem: str) -> Path:
    for ext in (".jpg", ".JPG", ".png"):
        for cand in (SRC / "images" / f"{stem}{ext}", *SRC.glob(f"images/**/{stem}{ext}")):
            if cand.exists():
                return cand
    raise FileNotFoundError(stem)


def starts(size: int) -> list[int]:
    s = list(range(0, max(size - TILE, 0) + 1, STRIDE))
    if s[-1] + TILE < size:
        s.append(size - TILE)
    return s


def main():
    names = [l.strip() for l in (SRC / "classes.txt").read_text().splitlines() if l.strip()]
    if OUT.exists():
        shutil.rmtree(OUT)
    stats = {}
    for split, listfile in (("train", "train_board.txt"), ("val", "val_board.txt")):
        (OUT / "images" / split).mkdir(parents=True)
        (OUT / "labels" / split).mkdir(parents=True)
        stems = [Path(l.strip()).stem for l in (SRC / "ImageSets" / listfile).read_text().splitlines() if l.strip()]
        pos = neg = boxes_total = 0
        for stem in stems:
            img = cv2.imread(str(find_image(stem)))
            H, W = img.shape[:2]
            boxes = []
            for line in find_label(stem).read_text().splitlines():
                if not line.strip():
                    continue
                c, cx, cy, bw, bh = line.split()
                cx, cy, bw, bh = float(cx) * W, float(cy) * H, float(bw) * W, float(bh) * H
                boxes.append((int(c), cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))
            boxes_total += len(boxes)
            for y0 in starts(H):
                for x0 in starts(W):
                    x1, y1 = x0 + TILE, y0 + TILE
                    lines = []
                    for c, a, b, cc, d in boxes:
                        ia, ib, ic, idd = max(a, x0), max(b, y0), min(cc, x1), min(d, y1)
                        if ic <= ia or idd <= ib:
                            continue
                        if (ic - ia) * (idd - ib) < MIN_KEEP * (cc - a) * (d - b):
                            continue
                        lines.append(f"{c} {((ia + ic) / 2 - x0) / TILE:.6f} {((ib + idd) / 2 - y0) / TILE:.6f} {(ic - ia) / TILE:.6f} {(idd - ib) / TILE:.6f}")
                    if not lines and random.random() > NEG_KEEP[split]:
                        continue
                    name = f"{stem}_{x0}_{y0}"
                    cv2.imwrite(str(OUT / "images" / split / f"{name}.jpg"), img[y0:y1, x0:x1], [cv2.IMWRITE_JPEG_QUALITY, 95])
                    (OUT / "labels" / split / f"{name}.txt").write_text("\n".join(lines))
                    pos += bool(lines); neg += not lines
        stats[split] = dict(images=len(stems), boxes=boxes_total, positive_tiles=pos, negative_tiles=neg)

    (OUT / "data.yaml").write_text(
        f"path: {OUT}\ntrain: images/train\nval: images/val\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)))
    for k, v in stats.items():
        print(k, v)
    print("classes:", names)


if __name__ == "__main__":
    main()
