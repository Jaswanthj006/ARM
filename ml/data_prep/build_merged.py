"""Build the merged training set: PKU tiles + DeepPCB + DsPCBSD+, all in the 6 BoardCheck classes.

Classes (same ids as ml/data/tiles/data.yaml):
  0 missing_hole  1 mouse_bite  2 open_circuit  3 short  4 spur  5 spurious_copper

Scale: the app tiles photos at PKU scale, so every source is resized until its median defect
size is close to the PKU tiles (open ~52 px ... spurious copper ~103 px, overall ~75 px).
  DeepPCB   median defect ~40 px at 640  -> resize x1.875 to 1200 px, cut 2x2 tiles of 640
  DsPCBSD+  median defect ~20 px at 226  -> resize to 640 (x2.83)

Class mapping:
  DeepPCB  1 open->2, 2 short->3, 3 mousebite->1, 4 spur->4, 5 copper->5,
           6 pin-hole -> erased: the box is filled with the aligned defect-free template,
           so the model never learns a real void as "background".
  DsPCBSD+ SH->3, SP->4, SC->5, OP->2, MB->1. Images that contain HB / CS / CFO / BMFO are left
           out (those defects are not in our 6 classes and must not be learned as background).
           Near-duplicate val images (dHash distance <= 4 to any train image) move to train,
           so the DsPCBSD+ val score is not inflated by leakage.

PKU tiles are linked (not copied) from ml/data/tiles.

Output: ml/data/merged/{images,labels}/{train,val_pku,val_deeppcb,val_dspcbsd}/ + data.yaml
Usage:  python3 -I ml/data_prep/build_merged.py
"""
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ML = Path(__file__).resolve().parent.parent
DATA = ML / "data"
OUT = DATA / "merged"
PKU = DATA / "tiles"
DPCB = DATA / "ext" / "deeppcb" / "PCBData"
DSP = DATA / "ext" / "dspcbsd" / "Data_YOLO"
NAMES = ["missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"]

DPCB_MAP = {1: 2, 2: 3, 3: 1, 4: 4, 5: 5}          # 6 = pin-hole, erased
DPCB_SIZE, DPCB_STARTS = 1200, (0, 560)
DSP_MAP = {0: 3, 1: 4, 2: 5, 3: 2, 4: 1}            # 5..8 = HB, CS, CFO, BMFO -> image skipped
MIN_KEEP = 0.6
rng = random.Random(0)

# Board looks for the black/white DeepPCB scans: (substrate BGR, copper BGR)
PALETTES = [
    ((20, 70, 10), (40, 150, 30)),       # green solder mask, darker gaps (like PKU)
    ((15, 55, 5), (60, 170, 70)),
    ((25, 30, 30), (90, 140, 200)),      # bare copper on dark laminate (like DsPCBSD+)
    ((40, 60, 70), (110, 160, 210)),
    ((60, 40, 20), (170, 140, 90)),      # blue mask
    ((200, 200, 200), (30, 30, 30)),     # original-like grey
]


def colourize(gray):
    """gray: 0 = copper, 255 = substrate (DeepPCB convention)."""
    sub, cu = (np.array(c, np.float32) for c in rng.choice(PALETTES))
    t = cv2.GaussianBlur(gray, (0, 0), rng.uniform(0.6, 1.6)).astype(np.float32)[..., None] / 255.0
    img = cu * (1 - t) + sub * t
    img *= rng.uniform(0.8, 1.2)
    img += np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(2, 8), img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def clip_boxes(boxes, x0, y0, size=640):
    lines = []
    for c, a, b, cc, d in boxes:
        ia, ib, ic, idd = max(a, x0), max(b, y0), min(cc, x0 + size), min(d, y0 + size)
        if ic <= ia or idd <= ib or (ic - ia) * (idd - ib) < MIN_KEEP * (cc - a) * (d - b):
            continue
        lines.append(f"{c} {((ia + ic) / 2 - x0) / size:.6f} {((ib + idd) / 2 - y0) / size:.6f} "
                     f"{(ic - ia) / size:.6f} {(idd - ib) / size:.6f}")
    return lines


def save(split, name, img, lines):
    cv2.imwrite(str(OUT / "images" / split / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    (OUT / "labels" / split / f"{name}.txt").write_text("\n".join(lines))


def link_pku(stats):
    for src, dst in (("train", "train"), ("val", "val_pku")):
        for img in sorted((PKU / "images" / src).glob("*.jpg")):
            lab = PKU / "labels" / src / f"{img.stem}.txt"
            (OUT / "images" / dst / f"pku_{img.name}").symlink_to(img)
            (OUT / "labels" / dst / f"pku_{img.stem}.txt").symlink_to(lab)
            stats[dst]["pku"] += 1


def build_deeppcb(stats, boxes_out):
    for listfile, split in (("trainval.txt", "train"), ("test.txt", "val_deeppcb")):
        for line in (DPCB / listfile).read_text().split("\n"):
            if not line.strip():
                continue
            img_rel, lab_rel = line.split()
            stem = Path(img_rel).stem
            folder = DPCB / Path(img_rel).parent
            test = cv2.imread(str(folder / f"{stem}_test.jpg"), cv2.IMREAD_GRAYSCALE)
            temp = cv2.imread(str(folder / f"{stem}_temp.jpg"), cv2.IMREAD_GRAYSCALE)
            if test is None or temp is None:
                stats["skipped"]["deeppcb_missing_file"] += 1
                continue
            boxes = []
            for l in (DPCB / lab_rel).read_text().split("\n"):
                p = l.replace(",", " ").split()
                if len(p) < 5:
                    continue
                x1, y1, x2, y2, t = map(int, p[:5])
                if t == 6:                              # pin-hole: paint the clean template back in
                    test[y1:y2, x1:x2] = temp[y1:y2, x1:x2]
                    stats["erased"]["deeppcb_pin_hole"] += 1
                elif t in DPCB_MAP:
                    boxes.append((DPCB_MAP[t], x1, y1, x2, y2))
            s = DPCB_SIZE / test.shape[1]
            big = cv2.resize(colourize(test), (DPCB_SIZE, DPCB_SIZE), interpolation=cv2.INTER_LINEAR)
            boxes = [(c, a * s, b * s, cc * s, d * s) for c, a, b, cc, d in boxes]
            for y0 in DPCB_STARTS:
                for x0 in DPCB_STARTS:
                    lines = clip_boxes(boxes, x0, y0)
                    save(split, f"dpcb_{stem}_{x0}_{y0}", big[y0:y0 + 640, x0:x0 + 640], lines)
                    stats[split]["deeppcb"] += 1
                    for ln in lines:
                        boxes_out[split][NAMES[int(ln.split()[0])]] += 1


def dhash(img, n=16):
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (n + 1, n))
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def build_dspcbsd(stats, boxes_out):
    items = {}
    for split in ("train", "val"):
        for img_path in sorted((DSP / "images" / split).glob("*.jpg")):
            lab = DSP / "labels" / split / f"{img_path.stem}.txt"
            rows = [l.split() for l in lab.read_text().split("\n") if l.strip()] if lab.exists() else []
            if any(int(r[0]) not in DSP_MAP for r in rows):
                stats["skipped"]["dspcbsd_other_defect_class"] += 1
                continue
            if not rows:
                stats["skipped"]["dspcbsd_no_label"] += 1
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                stats["skipped"]["dspcbsd_unreadable"] += 1
                continue
            items[(split, img_path.stem)] = (img, rows, dhash(img))
    train_hashes = [h for (sp, _), (_, _, h) in items.items() if sp == "train"]
    moved = 0
    for (split, stem), (img, rows, h) in items.items():
        dst = "train"
        if split == "val":
            if min((h ^ t).bit_count() for t in train_hashes) <= 4:
                moved += 1
            else:
                dst = "val_dspcbsd"
        big = cv2.resize(img, (640, 640), interpolation=cv2.INTER_CUBIC)
        lines = []
        for r in rows:
            c = DSP_MAP[int(r[0])]
            lines.append(f"{c} {' '.join(r[1:5])}")     # normalised coords survive the resize
            boxes_out[dst][NAMES[c]] += 1
        save(dst, f"dsp_{split}_{stem}", big, lines)
        stats[dst]["dspcbsd"] += 1
    stats["moved"]["dspcbsd_val_near_duplicates_to_train"] = moved


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    splits = ("train", "val_pku", "val_deeppcb", "val_dspcbsd")
    for sp in splits:
        (OUT / "images" / sp).mkdir(parents=True)
        (OUT / "labels" / sp).mkdir(parents=True)
    stats = {k: Counter() for k in (*splits, "skipped", "erased", "moved")}
    boxes_out = {k: Counter() for k in splits}
    link_pku(stats)
    build_deeppcb(stats, boxes_out)
    build_dspcbsd(stats, boxes_out)
    for sp in ("train", "val_pku"):           # PKU box counts
        for f in (OUT / "labels" / sp).glob("pku_*.txt"):
            for l in f.read_text().split("\n"):
                if l.strip():
                    boxes_out[sp][NAMES[int(l.split()[0])]] += 1
    (OUT / "data.yaml").write_text(
        f"path: {OUT}\ntrain: images/train\nval:\n  - images/val_pku\n  - images/val_deeppcb\n  - images/val_dspcbsd\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)))
    report = {"images": {k: dict(v) for k, v in stats.items()}, "boxes": {k: dict(v) for k, v in boxes_out.items()}}
    (ML / "results").mkdir(exist_ok=True)
    (ML / "results" / "merged_dataset.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
