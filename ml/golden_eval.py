"""Golden-board comparison: does it remove false alarms without hiding real defects?

Idea (same algorithm the app uses, ml/golden_eval.py <-> Golden.kt):
  The operator photographs one known-good board of the design once (the "golden" reference).
  For every detection on a new board, the reference is resized to the photo size, and the area
  around the detection is searched for in the reference (zero-mean normalised cross-correlation,
  so lighting changes do not matter). The search window allows the board to be shifted by
  SEARCH x the photo size. If the box itself also matches the reference closely (core ZNCC >= T),
  the "defect" is part of the design (e.g. a T-junction read as a short) and is suppressed.
  A real defect is not in the reference, so its core does not match and it is kept.

References used for testing (none of them is the tested photo):
  DefectDet / PCB-Defect: ANOTHER physically different board of the same design and side
      (e.g. 100_1_BOTTOM_B is the reference for 100_1_BOTTOM_A). Realistic: different board,
      different photo, slightly different size and framing, its own defects elsewhere.
  PKU: the pixel-wise median of the OTHER photos of the same board (defects are edited into
      different places, so the median is clean), then made imperfect like a real re-photograph:
      shifted up to 3 %, scaled 97-103 %, brightness +-15 %, slight blur.

Usage: python3 -I ml/golden_eval.py   (needs results/phase3_cache/v3.pkl from eval_v2.py)
Output: ml/results/golden_eval.json
"""
import json
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ML = Path(__file__).resolve().parent
sys.path.insert(0, str(ML))
import eval_v2 as E  # noqa: E402

THR = json.loads((ML / "results" / "phaseB_final.json").read_text())["final_thresholds"]
SEARCH = 0.04      # search +-4 % of the photo size around the expected place
PAD = 0.6          # context around the box used to find the place in the reference
WORK = 1000        # images are compared at this longest side (speed); the phone does the same
rng = random.Random(0)


def gray_small(img):
    s = WORK / max(img.shape[:2])
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return cv2.resize(g, (round(g.shape[1] * s), round(g.shape[0] * s)), interpolation=cv2.INTER_AREA), s


def core_match(test_g, ref_g, box, s):
    """Best ZNCC of the box (with context to locate it, then the core alone) inside the reference."""
    H, W = test_g.shape
    x1, y1, x2, y2 = [v * s for v in box]
    bw, bh = max(x2 - x1, 6), max(y2 - y1, 6)
    px, py = bw * PAD + 8, bh * PAD + 8
    cx1, cy1, cx2, cy2 = int(max(0, x1 - px)), int(max(0, y1 - py)), int(min(W, x2 + px)), int(min(H, y2 + py))
    patch = test_g[cy1:cy2, cx1:cx2]
    sx, sy = int(SEARCH * W), int(SEARCH * H)
    rx1, ry1, rx2, ry2 = max(0, cx1 - sx), max(0, cy1 - sy), min(ref_g.shape[1], cx2 + sx), min(ref_g.shape[0], cy2 + sy)
    region = ref_g[ry1:ry2, rx1:rx2]
    if region.shape[0] < patch.shape[0] or region.shape[1] < patch.shape[1] or patch.std() < 2:
        return 0.0
    res = cv2.matchTemplate(region, patch, cv2.TM_CCOEFF_NORMED)
    _, ctx_score, _, (mx, my) = cv2.minMaxLoc(res)
    # the box core at the matched place
    ox, oy = rx1 + mx + (int(x1) - cx1), ry1 + my + (int(y1) - cy1)
    core_t = test_g[int(y1):int(y1) + int(bh), int(x1):int(x1) + int(bw)]
    core_r = ref_g[oy:oy + core_t.shape[0], ox:ox + core_t.shape[1]]
    if core_r.shape != core_t.shape or core_t.size < 16:
        return 0.0
    a = core_t.astype(np.float32) - core_t.mean(); b = core_r.astype(np.float32) - core_r.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    core = float((a * b).sum() / den) if den > 1e-6 else (1.0 if core_t.std() < 3 and core_r.std() < 3 else 0.0)
    return core if ctx_score > 0.5 else 0.0          # place not found -> no evidence -> keep the detection


def imperfect(ref):
    H, W = ref.shape[:2]
    s = rng.uniform(0.97, 1.03); a = np.float32([[s, 0, rng.uniform(-0.03, 0.03) * W], [0, s, rng.uniform(-0.03, 0.03) * H]])
    out = cv2.warpAffine(ref, a, (W, H), borderMode=cv2.BORDER_REPLICATE)
    out = np.clip(out.astype(np.float32) * rng.uniform(0.85, 1.15), 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(out, (3, 3), 0.8)


def references():
    """name -> (test image path, reference image or path) for the three board sets."""
    refs = {}
    # DefectDet: same design + board number + side, another letter
    dd = json.loads((ML / "data/ext/defectdet/DefectDet/annotation/defect_annotations.json").read_text())
    orig = {Path(i["file_name"]).stem: i["extra"]["name"].rsplit(".", 1)[0] for i in dd["images"]}
    by_group = defaultdict(list)
    for stem, n in orig.items():
        by_group[n.rsplit("_", 1)[0]].append(stem)
    rt = ML / "data/real_test/defectdet/images"
    for p in rt.glob("*.jpg"):
        grp = orig[p.stem].rsplit("_", 1)[0]
        others = [o for o in sorted(by_group[grp]) if o != p.stem]
        if others:
            refs[("real_defectdet", p.stem)] = (p, ML / "data/ext/defectdet/DefectDet/images" / f"{others[0]}.png")
    # PCB-Defect: '56-1-3' -> another photo '56-1-x' of the same design group
    rt = ML / "data/real_test/pcbdefect/images"
    all_p = sorted((ML / "data/ext/pcbdefect/PCB-Defect-Reannotated/train/images").iterdir())
    for p in rt.glob("*.jpg"):
        grp = "-".join(p.stem.split("-")[:2])
        others = [o for o in all_p if "-".join(o.stem.split("-")[:2]) == grp and o.stem != p.stem]
        if others:
            refs[("real_pcbdefect", p.stem)] = (p, others[0])
    return refs


def main():
    rec = pickle.loads((ML / "results/phase3_cache/v3.pkl").read_bytes())
    refs = references()
    pku_median = {}
    pairs = {}
    for k, (tp, rp) in refs.items():
        pairs[k] = (cv2.imread(str(tp)), cv2.imread(str(rp)))
    # PKU half B with imperfect median references
    stems = sorted(rec["pku_boards"])
    for stem in stems[1::2]:
        b = stem.split("_")[0]
        if b not in pku_median:
            fs = sorted((ML / "data/pku/images").glob(f"{b}_*.jpg"))
            pku_median[b] = (fs, None)
        fs, _ = pku_median[b]
        others = [f for f in fs if f.stem != stem][::3][:9]
        med = np.median(np.stack([cv2.imread(str(f), 0) for f in others]), axis=0).astype(np.uint8)
        pairs[("pku_boards", stem)] = (cv2.imread(str(next(f for f in fs if f.stem == stem))), imperfect(med))
    print("pairs:", {s: sum(1 for k in pairs if k[0] == s) for s in ("pku_boards", "real_pcbdefect", "real_defectdet")}, flush=True)

    scores = {}           # (set, stem) -> list of core scores per detection kept by thresholds
    for (sname, stem), (img, ref) in pairs.items():
        r = rec[sname][stem]
        dets = [d for d in r["det"] if d[1] >= THR[E.NAMES[d[0]]]]
        tg, s = gray_small(img)
        ref = cv2.resize(ref if ref.ndim == 2 else cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), (img.shape[1], img.shape[0]))
        rg, _ = gray_small(ref)
        scores[(sname, stem)] = [core_match(tg, rg, d[2:], s) for d in dets]

    out = {"search": SEARCH, "pad": PAD, "work_px": WORK, "results": {}}
    for T in (None, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7):
        row = {}
        for sname in ("pku_boards", "real_pcbdefect", "real_defectdet"):
            keys = [k for k in pairs if k[0] == sname]
            sub = {}
            for k in keys:
                r = rec[sname][k[1]]
                dets = [d for d in r["det"] if d[1] >= THR[E.NAMES[d[0]]]]
                keep = dets if T is None else [d for d, c in zip(dets, scores[k]) if c < T]
                sub[k[1]] = dict(det=keep, gt=r["gt"])
            x = E.score(sub, {n: 0.0 for n in E.NAMES})
            row[sname] = dict(P=x["P"], R=x["R"], found=x["found"], missed=x["missed"], false=x["false"],
                              short_P=x["per_class"]["short"]["P"], short_R=x["per_class"]["short"]["R"])
        out["results"]["off" if T is None else f"T={T}"] = row
        print("golden off" if T is None else f"golden T={T}", " | ".join(
            f"{s} P {v['P']} R {v['R']} false {v['false']} missed {v['missed']} (short {v['short_P']}/{v['short_R']})" for s, v in row.items()), flush=True)
    (ML / "results/golden_eval.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
