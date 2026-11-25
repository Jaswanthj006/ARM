"""Add the two real-board datasets to the merged training set (7-class scheme).

Classes (0-5 unchanged, 6 new):
  0 missing_hole  1 mouse_bite  2 open_circuit  3 short  4 spur  5 spurious_copper  6 missing_pad

Sources (both CC BY 4.0, Mendeley Data)
  PCB-Defect v2 (re-annotated, oriented boxes -> axis-aligned), 230 photos of 40 board designs.
      Rashid et al., Data in Brief 64 (2026) 112296, doi:10.1016/j.dib.2025.112296
  DefectDet, 268 photos of real manufactured double-layer boards, COCO boxes. doi:10.17632/t9d9zs3bmb

Split: ~20 % of board DESIGNS (PCB-Defect: '50-1' from '50-1-3.jpg'; DefectDet: board number from
'22_2_BOTTOM_H.png') are held out for testing; no design appears in both train and test.
Defects in both sets are already about PKU size (median 40-110 px), so photos are tiled at native
resolution: 640 px tiles, stride 480, a box is kept in a tile if >= 60 % of it lies inside, and 12 %
of defect-free tiles are kept as clean examples (same rules as PKU in prepare_tiles.py).

Output
  ml/data/merged/{images,labels}/train/{pcbd,ddet}_*.jpg       training tiles
  ml/data/merged/{images,labels}/val_{pcbdefect,defectdet}/    test tiles (for Ultralytics val)
  ml/data/real_test/{pcbdefect,defectdet}/{images,labels}/     whole test boards (for app-style tests)
  ml/results/real_boards_dataset.json                           counts
Usage: python3 -I ml/data_prep/add_real_boards.py
"""
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2

ML = Path(__file__).resolve().parent.parent
EXT = ML / "data" / "ext"
M = ML / "data" / "merged"
RT = ML / "data" / "real_test"
NAMES = ["missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper", "missing_pad"]
TILE, STRIDE, MIN_KEEP, NEG_KEEP, TEST_SHARE = 640, 480, 0.6, 0.12, 0.2
PCBD_MAP = {0: 6, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}                     # PCB-Defect data.yaml order
DDET_MAP = {"MP": 6, "OC": 2, "SC": 3, "SP": 4, "SPC": 5}            # DefectDet category names
rng = random.Random(0)


def starts(size):
    s = list(range(0, max(size - TILE, 0) + 1, STRIDE))
    if s[-1] + TILE < size:
        s.append(size - TILE)
    return s


def load_pcbdefect():
    root = EXT / "pcbdefect" / "PCB-Defect-Reannotated" / "train"
    items = []
    for img in sorted((root / "images").iterdir()):
        if img.name.startswith(".") or img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        boxes = []
        for l in (root / "labels" / f"{img.stem}.txt").read_text().split("\n"):
            p = l.split()
            if len(p) < 9:
                continue
            xs, ys = [float(v) for v in p[1:9:2]], [float(v) for v in p[2:9:2]]
            boxes.append((PCBD_MAP[int(p[0])], min(xs), min(ys), max(xs), max(ys)))   # normalised
        items.append(dict(src="pcbd", path=img, group="-".join(img.stem.split("-")[:2]), boxes=boxes, norm=True))
    return items


def load_defectdet():
    root = EXT / "defectdet" / "DefectDet"
    d = json.loads((root / "annotation" / "defect_annotations.json").read_text())
    cats = {c["id"]: c["name"] for c in d["categories"]}
    per = defaultdict(list)
    for a in d["annotations"]:
        x, y, w, h = a["bbox"]
        per[a["image_id"]].append((DDET_MAP[cats[a["category_id"]]], x, y, x + w, y + h))
    items = []
    for im in d["images"]:
        orig = (im.get("extra") or {}).get("name", im["file_name"])
        items.append(dict(src="ddet", path=root / "images" / im["file_name"], group=orig.split("_")[0],
                          boxes=per.get(im["id"], []), norm=False))
    return items


def split(items):
    groups = sorted({it["group"] for it in items})
    rng.shuffle(groups)
    test = set(groups[: max(1, round(TEST_SHARE * len(groups)))])
    return [i for i in items if i["group"] not in test], [i for i in items if i["group"] in test], sorted(test)


def tile(item, img, split_dir, stats, boxcount):
    H, W = img.shape[:2]
    boxes = [(c, a * W, b * H, cc * W, d * H) if item["norm"] else (c, a, b, cc, d) for c, a, b, cc, d in item["boxes"]]
    stem = f"{item['src']}_{item['path'].stem}"
    for y0 in starts(H):
        for x0 in starts(W):
            lines = []
            for c, a, b, cc, d in boxes:
                ia, ib, ic, idd = max(a, x0), max(b, y0), min(cc, x0 + TILE), min(d, y0 + TILE)
                if ic <= ia or idd <= ib or (ic - ia) * (idd - ib) < MIN_KEEP * (cc - a) * (d - b):
                    continue
                lines.append(f"{c} {((ia + ic) / 2 - x0) / TILE:.6f} {((ib + idd) / 2 - y0) / TILE:.6f} "
                             f"{(ic - ia) / TILE:.6f} {(idd - ib) / TILE:.6f}")
            if not lines and rng.random() > NEG_KEEP:
                continue
            crop = img[y0:y0 + TILE, x0:x0 + TILE]
            if crop.shape[0] < TILE or crop.shape[1] < TILE:   # image smaller than a tile: pad like the app's letterbox
                crop = cv2.copyMakeBorder(crop, 0, TILE - crop.shape[0], 0, TILE - crop.shape[1], cv2.BORDER_CONSTANT, value=(114, 114, 114))
            name = f"{stem}_{x0}_{y0}"
            cv2.imwrite(str(M / "images" / split_dir / f"{name}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
            (M / "labels" / split_dir / f"{name}.txt").write_text("\n".join(lines))
            stats[split_dir][item["src"] + ("_pos" if lines else "_neg")] += 1
            for l in lines:
                boxcount[split_dir][NAMES[int(l.split()[0])]] += 1


def save_board(item, img, folder):
    H, W = img.shape[:2]
    (folder / "images").mkdir(parents=True, exist_ok=True); (folder / "labels").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(folder / "images" / f"{item['path'].stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    rows = []
    for c, a, b, cc, d in item["boxes"]:
        if not item["norm"]:
            a, cc, b, d = a / W, cc / W, b / H, d / H
        rows.append(f"{c} {(a + cc) / 2:.6f} {(b + d) / 2:.6f} {cc - a:.6f} {d - b:.6f}")
    (folder / "labels" / f"{item['path'].stem}.txt").write_text("\n".join(rows))


def main():
    for d in ("val_pcbdefect", "val_defectdet"):
        for k in ("images", "labels"):
            shutil.rmtree(M / k / d, ignore_errors=True); (M / k / d).mkdir(parents=True)
    for k in ("images", "labels"):
        for p in (M / k / "train").glob("pcbd_*"):
            p.unlink()
        for p in (M / k / "train").glob("ddet_*"):
            p.unlink()
    shutil.rmtree(RT, ignore_errors=True)
    stats, boxcount, report = defaultdict(Counter), defaultdict(Counter), {}
    for name, items, valdir in (("pcbdefect", load_pcbdefect(), "val_pcbdefect"), ("defectdet", load_defectdet(), "val_defectdet")):
        tr, te, test_groups = split(items)
        report[name] = dict(images=len(items), train_images=len(tr), test_images=len(te), test_designs=test_groups)
        for it in tr:
            tile(it, cv2.imread(str(it["path"])), "train", stats, boxcount)
        for it in te:
            img = cv2.imread(str(it["path"]))
            tile(it, img, valdir, stats, boxcount)
            save_board(it, img, RT / name)
        print(f"{name}: {len(tr)} train / {len(te)} test photos, test designs {test_groups}", flush=True)
    report["tiles"] = {k: dict(v) for k, v in stats.items()}
    report["boxes"] = {k: dict(v) for k, v in boxcount.items()}
    report["classes"] = NAMES
    (ML / "results" / "real_boards_dataset.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
