"""Phase C: export the v3 detector (7 classes) to ExecuTorch FP16 for the phone and check it.

Same recipe as export_yolo_fp16.py (XNNPACK, FP16 compute inside the graph, FP32 input/output,
ExecuTorch 1.6.0) so the app's patched runtime and threading are unchanged.
Model output: (1, 4 + 7, 8400) = cx, cy, w, h in 640 px input space, then 7 class scores.

Check: the .pte (ExecuTorch runtime) and the PyTorch FP32 model are run on the same test tiles
(PKU boards 10-12, PCB-Defect and DefectDet held-out designs). Detections are decoded with the
final per-class thresholds and scored against ground truth (IoU >= 0.5), and raw score differences
are reported. Mac CPU time per tile is measured too (phone timings come from the app).

Usage:  ml/.venv-et/bin/python -I ml/export_v3.py [--per-set 100]
Output: ml/models/pcb_yolo11n_v3_fp16.pte, ml/results/export_v3.json
"""
import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import torch

ML = Path(__file__).resolve().parent
sys.path.insert(0, str(ML))
from export_yolo import decode, detector, load_tile, lower  # noqa: E402
from export_yolo_fp16 import Half  # noqa: E402

from executorch.runtime import Runtime  # noqa: E402

NAMES = ["missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper", "missing_pad"]
FINAL = json.loads((ML / "results" / "phaseB_final.json").read_text())["final_thresholds"]
M = ML / "data" / "merged"


def gt(lab):
    out = []
    for l in lab.read_text().split("\n"):
        if l.strip():
            c, x, y, w, h = map(float, l.split()[:5])
            out.append((int(c), (x - w / 2) * 640, (y - h / 2) * 640, (x + w / 2) * 640, (y + h / 2) * 640))
    return out


def keep(dets):
    return [d for d in dets if d[1] >= FINAL[NAMES[d[0]]]]


def match(dets, gts):
    tp = fp = 0; used = set()
    for c, s, *b in sorted(dets, key=lambda d: -d[1]):
        best, bj = 0.0, None
        for j, (gc, *g) in enumerate(gts):
            if j in used or gc != c:
                continue
            ix = max(0, min(b[2], g[2]) - max(b[0], g[0])); iy = max(0, min(b[3], g[3]) - max(b[1], g[1]))
            inter = ix * iy; u = (b[2] - b[0]) * (b[3] - b[1]) + (g[2] - g[0]) * (g[3] - g[1]) - inter
            if u > 0 and inter / u > best:
                best, bj = inter / u, j
        if best >= 0.5:
            used.add(bj); tp += 1
        else:
            fp += 1
    return tp, fp, len(gts) - len(used)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ML / "runs/pcb_yolo11n_v3/weights/best.pt"))
    ap.add_argument("--out", default=str(ML / "models/pcb_yolo11n_v3_fp16.pte"))
    ap.add_argument("--per-set", type=int, default=100)
    a = ap.parse_args()

    m = detector(a.weights)
    x = torch.rand(1, 3, 640, 640)
    with torch.no_grad():
        shape = tuple((m(x)[0] if isinstance(m(x), (tuple, list)) else m(x)).shape)
    print("PyTorch output shape:", shape)
    buf = lower(torch.export.export(Half(copy.deepcopy(m)).eval(), (x,)))   # Half() converts in place: keep m FP32
    Path(a.out).write_bytes(buf)
    print("saved", a.out, round(len(buf) / 1e6, 2), "MB")

    method = Runtime.get().load_program(buf).load_method("forward")
    rng = random.Random(0)
    report = dict(model=a.weights, pte=a.out, size_mb=round(len(buf) / 1e6, 2), output_shape=shape,
                  thresholds=FINAL, sets={})
    tot = {"pytorch": [0, 0, 0], "pte": [0, 0, 0]}
    max_score_diff, times = 0.0, []
    for split in ("val_pku", "val_pcbdefect", "val_defectdet"):
        imgs = sorted(p for p in (M / "images" / split).glob("*.jpg") if not p.name.startswith("."))
        imgs = rng.sample(imgs, min(a.per_set, len(imgs)))
        res = {"pytorch": [0, 0, 0], "pte": [0, 0, 0]}
        for p in imgs:
            t = load_tile(p)
            g = gt(M / "labels" / split / f"{p.stem}.txt")
            with torch.no_grad():
                r = m(t); r = r[0] if isinstance(r, (tuple, list)) else r
            t0 = time.perf_counter(); y = method.execute([t])[0]; times.append(time.perf_counter() - t0)
            max_score_diff = max(max_score_diff, float((y[:, 4:] - r[:, 4:]).abs().max()))
            for k, out in (("pytorch", r), ("pte", y)):
                tp, fp, fn = match(keep(decode(out, conf=0.2)), g)
                for i, v in enumerate((tp, fp, fn)):
                    res[k][i] += v; tot[k][i] += v
        report["sets"][split] = {k: dict(found=v[0], false=v[1], missed=v[2],
                                         P=round(v[0] / max(v[0] + v[1], 1), 3), R=round(v[0] / max(v[0] + v[2], 1), 3))
                                 for k, v in res.items()}
        print(split, report["sets"][split], flush=True)
    report["all"] = {k: dict(P=round(v[0] / max(v[0] + v[1], 1), 3), R=round(v[0] / max(v[0] + v[2], 1), 3)) for k, v in tot.items()}
    report["max_class_score_diff_fp16_vs_fp32"] = round(max_score_diff, 4)
    report["mac_cpu_ms_per_tile_median"] = round(1000 * sorted(times)[len(times) // 2], 1)
    (ML / "results" / "export_v3.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("size_mb", "output_shape", "all", "max_class_score_diff_fp16_vs_fp32", "mac_cpu_ms_per_tile_median")}, indent=1))


if __name__ == "__main__":
    main()
