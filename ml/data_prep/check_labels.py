"""Draw labels on random merged-set images, one contact sheet per source, for a by-eye check.

Usage:  python3 -I ml/data_prep/check_labels.py <out_dir> [n_per_sheet]
"""
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ML = Path(__file__).resolve().parent.parent
M = ML / "data" / "merged"
NAMES = ["missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"]
SHORT = ["MH", "MB", "OP", "SH", "SP", "SC", "MP"]
COL = [(255, 0, 255), (0, 255, 255), (0, 0, 255), (255, 255, 0), (0, 165, 255), (255, 0, 0), (0, 255, 0)]


def sheet(paths, out, cell=320):
    tiles = []
    for p in paths:
        img = cv2.imread(str(p))
        lab = M / "labels" / p.parent.name / f"{p.stem}.txt"
        h, w = img.shape[:2]
        for l in lab.read_text().split("\n"):
            if not l.strip():
                continue
            c, x, y, bw, bh = l.split()
            c = int(c)
            x1, y1 = int((float(x) - float(bw) / 2) * w), int((float(y) - float(bh) / 2) * h)
            x2, y2 = int((float(x) + float(bw) / 2) * w), int((float(y) + float(bh) / 2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), COL[c], 3)
            cv2.putText(img, SHORT[c], (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, COL[c], 3)
        tiles.append(cv2.resize(img, (cell, cell)))
    while len(tiles) % 4:
        tiles.append(np.zeros((cell, cell, 3), np.uint8))
    cv2.imwrite(str(out), np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]), [cv2.IMWRITE_JPEG_QUALITY, 85])


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    rng = random.Random(1)
    for prefix in (sys.argv[3].split(",") if len(sys.argv) > 3 else ("pku", "dpcb", "dsp")):
        imgs = [p for p in (M / "images" / "train").glob(f"{prefix}_*.jpg")
                if (M / "labels" / "train" / f"{p.stem}.txt").read_text().strip()]
        sheet(rng.sample(imgs, n), out / f"labels_{prefix}.jpg")
        print("wrote", out / f"labels_{prefix}.jpg")


if __name__ == "__main__":
    main()
