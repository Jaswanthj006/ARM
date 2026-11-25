"""Board masks for the real phone frames (ml/data/pcb_real/phone), used only as training
labels for the gate's board map. The boards were filmed on a light cloth, so the board is
the largest dark region; its fitted rectangle is the mask.

Usage: python3 ml/real_board_masks.py [preview.jpg]   -> ml/data/pcb_real/masks/<name>.png
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
SRC, OUT = ROOT / "data/pcb_real/phone", ROOT / "data/pcb_real/masks"


def mask_for(img):
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    _, m = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)   # dark = board
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((81, 1), np.uint8))   # cloth stripes are horizontal, the board is tall
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return None
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    cnt = cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]
    out = np.zeros(m.shape, np.uint8)
    cv2.fillPoly(out, [cv2.boxPoints(cv2.minAreaRect(cnt)).astype(np.int32)], 255)   # boards are rectangles
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tiles = []
    for k, f in enumerate(sorted(SRC.glob("*.jpg"))):
        img = cv2.imread(str(f)); m = mask_for(img)
        if m is None:
            continue
        cv2.imwrite(str(OUT / (f.stem + ".png")), m)
        if k % 7 == 0 and len(tiles) < 12:
            v = img.copy(); v[m == 0] = (v[m == 0] * 0.35).astype(np.uint8)
            tiles.append(cv2.resize(v, (240, 320)))
    print("masks:", len(list(OUT.glob("*.png"))))
    if len(sys.argv) > 1 and len(tiles) == 12:
        cv2.imwrite(sys.argv[1], np.vstack([np.hstack(tiles[:6]), np.hstack(tiles[6:])]))


if __name__ == "__main__":
    main()
