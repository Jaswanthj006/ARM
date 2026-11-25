"""Cut the board out of handheld phone videos -> real clean-board crops (negatives / false-alarm test).

For each video frame: colour mask for the board's solder mask (red or green), largest blob, bounding box
padded by 4 %. A frame is kept only if the board is fully inside the frame, the blob fills its box
(rectangular board, not a hand-covered fragment), the crop is sharp, and it is >= STEP frames after the
last kept frame (no near-duplicates). The first 70 % of each video's time is 'train', the rest 'test'.

Usage:
  python3 -I ml/data_prep/extract_video_boards.py <video> <board_id> <red|green> [<video> <board_id> <colour> ...]
Output: ml/data/real_video/{train,test}/<board_id>_<frame>.jpg + empty YOLO labels, manifest.json
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ML = Path(__file__).resolve().parent.parent
OUT = ML / "data" / "real_video"
STEP, MIN_SHARP, MIN_FILL, MIN_W, PAD, TRAIN_SHARE = 6, 150.0, 0.80, 220, 0.04, 0.70


def board_mask(img, colour):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if colour == "red":
        m = cv2.inRange(hsv, (0, 120, 70), (8, 255, 255)) | cv2.inRange(hsv, (170, 120, 70), (180, 255, 255))
    else:
        m = cv2.inRange(hsv, (40, 80, 40), (90, 255, 255))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))


def main():
    args = sys.argv[1:]
    if not args or len(args) % 3:
        sys.exit(__doc__)
    for sp in ("train", "test"):
        (OUT / sp).mkdir(parents=True, exist_ok=True)
    manifest = []
    for video, board, colour in zip(args[0::3], args[1::3], args[2::3]):
        cap = cv2.VideoCapture(video)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        last, i, kept, why = -STEP, -1, 0, {"edge": 0, "fill": 0, "small": 0, "blur": 0, "dup": 0}
        while True:
            ok, img = cap.read()
            i += 1
            if not ok:
                break
            H, W = img.shape[:2]
            cs, _ = cv2.findContours(board_mask(img, colour), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cs:
                continue
            c = max(cs, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            if x <= 2 or y <= 2 or x + w >= W - 2 or y + h >= H - 2:
                why["edge"] += 1; continue
            if w < MIN_W:
                why["small"] += 1; continue
            if cv2.contourArea(c) / (w * h) < MIN_FILL:
                why["fill"] += 1; continue
            if i - last < STEP:
                why["dup"] += 1; continue
            p = int(PAD * max(w, h))
            x0, y0, x1, y1 = max(0, x - p), max(0, y - p), min(W, x + w + p), min(H, y + h + p)
            crop = img[y0:y1, x0:x1]
            sharp = cv2.Laplacian(cv2.cvtColor(img[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            if sharp < MIN_SHARP:
                why["blur"] += 1; continue
            sp = "train" if i < TRAIN_SHARE * n else "test"
            name = f"{board}_{i:04d}"
            cv2.imwrite(str(OUT / sp / f"{name}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            (OUT / sp / f"{name}.txt").write_text("")
            manifest.append(dict(file=f"{sp}/{name}.jpg", board=board, colour=colour, frame=i, split=sp,
                                 board_px=[w, h], sharpness=round(sharp, 1), source=Path(video).name))
            last, kept = i, kept + 1
        print(f"{board}: {n} frames -> kept {kept}; rejected {why}")
    (OUT / "manifest.json").write_text(json.dumps(dict(
        note="Clean boards (no labelled defects). Handheld WhatsApp video, 478x850 px; board ~8-10 px/mm, "
             "3-5x below training scale, so use as negatives / false-alarm test, not as defect positives.",
        crops=manifest), indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
