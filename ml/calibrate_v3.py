"""Confidence calibration for the v3 detector (7 classes): raw score -> fraction really correct.

Uses the v3 detections already computed by eval_v2.py (results/phase3_cache/v3.pkl, whole boards tiled
like the app) on PKU boards 10-12 and the held-out real boards (PCB-Defect, DefectDet). A detection is
correct if it overlaps a labelled defect of the same class (IoU >= 0.5).

Per defect type: monotonic curve (isotonic regression) on a score grid, pooled with all types when a type
has fewer than 40 detections. Honest check: fit on half A of the photos, report reliability on half B;
the curves written for the website are then fitted on all photos.

Writes web/js/calibration.json (same format the website reads; the old file is kept as
calibration_v2_old.json) and ml/results/calibration_v3.json.
Usage: python3 -I ml/calibrate_v3.py
"""
import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

ML = Path(__file__).resolve().parent
sys.path.insert(0, str(ML))
import eval_v2 as E  # noqa: E402

GRID = [round(x, 2) for x in np.arange(0.20, 1.0001, 0.05)]
MIN_SCORE = 0.20
SETS = ("pku_boards", "real_pcbdefect", "real_defectdet")
BINS = [(0, 0.5, "0-50%"), (0.5, 0.7, "50-70%"), (0.7, 0.8, "70-80%"), (0.8, 0.9, "80-90%"), (0.9, 1.01, "90-100%")]


def collect(rec):
    rows = []    # (key, type, score, correct)
    for s in SETS:
        for stem, r in rec[s].items():
            used = set()
            for c, sc, *b in sorted(r["det"], key=lambda d: -d[1]):
                if sc < MIN_SCORE:
                    continue
                best, bj = 0.0, None
                for j, g in enumerate(r["gt"]):
                    if j in used or g[0] != c:
                        continue
                    v = E.iou(b, g[1:])
                    if v > best:
                        best, bj = v, j
                ok = best >= 0.5
                if ok:
                    used.add(bj)
                rows.append((f"{s}/{stem}", E.NAMES[c], sc, ok))
    return rows


def fit(rows):
    def iso(sub):
        x = np.array([r[2] for r in sub]); y = np.array([r[3] for r in sub], float)
        m = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(x, y)
        return [round(float(v), 3) for v in m.predict(GRID)]
    pooled = iso(rows)
    curves = {}
    for t in E.NAMES:
        sub = [r for r in rows if r[1] == t]
        curves[t] = iso(sub) if len(sub) >= 40 else pooled
    curves["open"] = curves["open_circuit"]
    return curves


def cal(curves, t, s):
    return float(np.interp(s, GRID, curves[t]))


def reliability(rows, curves):
    out, err_raw, err_cal, n = [], 0.0, 0.0, len(rows)
    for lo, hi, name in BINS:
        b = [r for r in rows if lo <= cal(curves, r[1], r[2]) < hi]
        if b:
            out.append(dict(stated=name, detections=len(b), avg_stated=round(np.mean([cal(curves, r[1], r[2]) for r in b]), 3),
                            actually_correct=round(np.mean([r[3] for r in b]), 3)))
    for key in ("raw", "cal"):
        f = (lambda r: r[2]) if key == "raw" else (lambda r: cal(curves, r[1], r[2]))
        e = 0.0
        for lo, hi, _ in [(i / 10, (i + 1) / 10 + (0.01 if i == 9 else 0), "") for i in range(10)]:
            b = [r for r in rows if lo <= f(r) < hi]
            if b:
                e += len(b) / n * abs(np.mean([f(r) for r in b]) - np.mean([r[3] for r in b]))
        if key == "raw":
            err_raw = e
        else:
            err_cal = e
    return out, round(err_raw, 3), round(err_cal, 3)


def main():
    rec = pickle.loads((ML / "results/phase3_cache/v3.pkl").read_bytes())
    rows = collect(rec)
    keys = sorted({r[0] for r in rows})
    half_a = set(keys[0::2])
    a = [r for r in rows if r[0] in half_a]; b = [r for r in rows if r[0] not in half_a]
    rel, e_raw, e_cal = reliability(b, fit(a))
    curves = fit(rows)
    boards = len({k for s in SETS for k in rec[s]})
    report = dict(boards=boards, detections=len(rows), min_score=MIN_SCORE, max_side="full photo, 640 px tiles",
                  calibration_error_raw_score=e_raw, calibration_error_calibrated=e_cal,
                  reliability_held_out_half=rel,
                  per_type={t: dict(detections=sum(r[1] == t for r in rows), correct=int(sum(r[3] for r in rows if r[1] == t))) for t in E.NAMES},
                  note="v3 detector. Calibrated on PKU boards 10-12 and held-out real boards (PCB-Defect, DefectDet); "
                       "reliability measured on the half of the photos not used for fitting. Real phone photos may differ.")
    web = ML.parent / "web/js/calibration.json"
    old = ML.parent / "web/js/calibration_v2_old.json"
    if web.exists() and not old.exists():
        shutil.copy2(web, old)
    web.write_text(json.dumps(dict(grid=GRID, curves=curves, report=report)))
    (ML / "results/calibration_v3.json").write_text(json.dumps(dict(report=report, curves=curves), indent=1))
    print(json.dumps(report, indent=1))
    print("short curve:", dict(zip(GRID, curves["short"])))


if __name__ == "__main__":
    main()
