"""Export the trained PCB detector to ExecuTorch (XNNPACK, Arm CPU) as FP32 and INT8.

INT8 uses PT2E post-training quantization with the XNNPACK quantizer
(symmetric, per-channel weights). Calibration uses training tiles only, so the
validation boards stay unseen.

Then both .pte files are run through the ExecuTorch runtime on validation tiles
and scored against ground truth (IoU >= 0.5), so the cost of quantization is
measured, not assumed.

Usage:  ml/.venv-et/bin/python ml/export_yolo.py [--calib 150] [--eval 300]
Output: ml/models/pcb_yolo11n_fp32.pte, ml/models/pcb_yolo11n_int8.pte,
        ml/results/yolo_export.json
Model output: (1, 4 + 6, 8400) = cx, cy, w, h in 640 px input space, then class scores.
"""
import argparse
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from ultralytics import YOLO

from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer, get_symmetric_quantization_config)
from executorch.exir import to_edge_transform_and_lower
from executorch.runtime import Runtime
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

ROOT = Path(__file__).resolve().parent
TILES = ROOT / "data" / "tiles"
OUT = ROOT / "models"
IMG = 640


def load_tile(path):
    """Same preprocessing the phone app uses: BGR->RGB, /255, CHW float32."""
    im = cv2.imread(str(path))[:, :, ::-1]
    return torch.from_numpy(np.ascontiguousarray(im.transpose(2, 0, 1))).float().div(255).unsqueeze(0)


def detector(weights):
    model = YOLO(weights).model.float().eval().fuse()
    for m in model.modules():
        if m.__class__.__name__ == "Detect":
            m.export = True
            m.format = "executorch"
    return model


def lower(ep):
    return to_edge_transform_and_lower(ep, partitioner=[XnnpackPartitioner()]).to_executorch().buffer


def decode(out, conf=0.25, iou=0.45):
    """(1, 10, 8400) -> list of (cls, score, x1, y1, x2, y2). Mirrors the Kotlin decoder."""
    p = out[0].T  # 8400 x 10
    scores, cls = p[:, 4:].max(1)
    keep = scores >= conf
    if not keep.any():
        return []
    b, s, c = p[keep, :4], scores[keep], cls[keep]
    xyxy = torch.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)
    k = torchvision.ops.batched_nms(xyxy, s, c, iou)
    return [(int(c[i]), float(s[i]), *xyxy[i].tolist()) for i in k]


def score(dets_per_tile, labels_per_tile, ncls):
    tp = fp = fn = 0
    for dets, gts in zip(dets_per_tile, labels_per_tile):
        used = set()
        for c, s, *bx in sorted(dets, key=lambda d: -d[1]):
            best, bj = 0.0, None
            for j, (gc, *gb) in enumerate(gts):
                if j in used or gc != c:
                    continue
                ix = max(0, min(bx[2], gb[2]) - max(bx[0], gb[0])); iy = max(0, min(bx[3], gb[3]) - max(bx[1], gb[1]))
                inter = ix * iy
                u = (bx[2] - bx[0]) * (bx[3] - bx[1]) + (gb[2] - gb[0]) * (gb[3] - gb[1]) - inter
                v = inter / u if u > 0 else 0
                if v > best:
                    best, bj = v, j
            if best >= 0.5:
                used.add(bj); tp += 1
            else:
                fp += 1
        fn += len(gts) - len(used)
    return dict(precision=round(tp / max(tp + fp, 1), 3), recall=round(tp / max(tp + fn, 1), 3), found=tp, missed=fn, false_alarms=fp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "runs/pcb_yolo11n/weights/best.pt"))
    ap.add_argument("--calib", type=int, default=150)
    ap.add_argument("--eval", type=int, default=300)
    a = ap.parse_args()
    random.seed(0)
    OUT.mkdir(exist_ok=True)
    x = torch.zeros(1, 3, IMG, IMG)

    # FP32
    model = detector(a.weights)
    t = time.time()
    fp32 = lower(torch.export.export(model, (x,)))
    (OUT / "pcb_yolo11n_fp32.pte").write_bytes(fp32)
    print(f"FP32: {len(fp32) / 1e6:.1f} MB in {time.time() - t:.0f}s")

    # INT8 (PT2E, XNNPACK quantizer, calibrated on training tiles)
    t = time.time()
    gm = torch.export.export(detector(a.weights), (x,)).module()
    # The Detect head (model.23) concatenates box coordinates (0-640) and class
    # probabilities (0-1) into one tensor; a shared int8 scale crushes the
    # probabilities to zero. Keep the head in FP32, quantize backbone and neck.
    def not_head(node):
        stack = node.meta.get("nn_module_stack", {})
        return not any(str(path).startswith(("model.23", "L['self'].model[23]", "L__self___model_23"))
                       or ".23." in str(path) or str(path).endswith(".23")
                       for path, _ in stack.values())
    q = (XNNPACKQuantizer().set_global(get_symmetric_quantization_config(is_per_channel=True))
         .set_filter_function(not_head))
    prepared = prepare_pt2e(gm, q)
    calib = random.sample(sorted((TILES / "images/train").glob("*.jpg")), a.calib)
    with torch.no_grad():
        for p in calib:
            prepared(load_tile(p))
    int8 = lower(torch.export.export(convert_pt2e(prepared), (x,)))
    (OUT / "pcb_yolo11n_int8.pte").write_bytes(int8)
    print(f"INT8: {len(int8) / 1e6:.1f} MB in {time.time() - t:.0f}s (calibrated on {a.calib} training tiles)")

    # Score both .pte files with the ExecuTorch runtime on validation tiles (unseen boards).
    val = sorted((TILES / "images/val").glob("*.jpg"))
    val = random.sample(val, min(a.eval, len(val)))
    labels = []
    for p in val:
        gts = []
        for line in (TILES / "labels/val" / f"{p.stem}.txt").read_text().splitlines():
            if line.strip():
                c, cx, cy, w, h = map(float, line.split())
                gts.append((int(c), (cx - w / 2) * IMG, (cy - h / 2) * IMG, (cx + w / 2) * IMG, (cy + h / 2) * IMG))
        labels.append(gts)
    rt = Runtime.get()
    results = {}
    with torch.no_grad():
        ref = [decode(model(load_tile(p))) for p in val]
    results["pytorch_fp32"] = score(ref, labels, 6)
    for name in ("fp32", "int8"):
        prog = rt.load_program(str(OUT / f"pcb_yolo11n_{name}.pte"))
        method = prog.load_method("forward")
        dets, times = [], []
        for p in val:
            inp = load_tile(p)
            t = time.perf_counter(); out = method.execute([inp])[0]; times.append(time.perf_counter() - t)
            dets.append(decode(out))
        r = score(dets, labels, 6)
        r["mac_cpu_ms_per_tile"] = round(1000 * float(np.median(times)), 1)
        r["size_mb"] = round((OUT / f"pcb_yolo11n_{name}.pte").stat().st_size / 1e6, 1)
        results[f"executorch_{name}"] = r
    summary = dict(val_tiles=len(val), conf=0.25, iou_match=0.5, results=results,
                   note="Tile-level check on unseen boards 10-12; Mac CPU timings, phone timings come from the app.")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "yolo_export.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
