"""Export the PCB detector with FP16 compute for the phone (XNNPACK native half-precision).

Input and output stay FP32 (the app code is unchanged); the network runs in FP16
inside the graph, which Arm v8.2+ cores (A715/A510) execute natively.

Usage:  ml/.venv-et/bin/python ml/export_yolo_fp16.py
Output: ml/models/pcb_yolo11n_fp16.pte
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_yolo import detector, lower  # noqa: E402

ROOT = Path(__file__).resolve().parent


class Half(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m.half()

    def forward(self, x):
        return self.m(x.half()).float()


def main():
    m = detector(str(ROOT / "runs/pcb_yolo11n/weights/best.pt"))
    x = torch.rand(1, 3, 640, 640)
    with torch.no_grad():
        ref = m(x)
    h = Half(m).eval()
    buf = lower(torch.export.export(h, (x,)))
    out = ROOT / "models/pcb_yolo11n_fp16.pte"
    out.write_bytes(buf)
    print("saved", out, round(len(buf) / 1e6, 2), "MB")
    try:
        from executorch.runtime import Runtime
        prog = Runtime.get().load_program(buf)
        y = prog.load_method("forward").execute([x])[0]
        r = ref[0] if isinstance(ref, (tuple, list)) else ref
        print("max |fp16 - fp32| on box/score outputs:", float((y - r).abs().max()),
              "| score max diff:", float((y[:, 4:] - r[:, 4:]).abs().max()))
    except Exception as e:
        print("runtime check skipped:", e)


if __name__ == "__main__":
    main()
