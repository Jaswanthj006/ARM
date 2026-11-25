"""Phone end-to-end test for the v3 app: phone app -> bridge -> this script (acting as the website).

For each test board stored on the phone (files/test/), triggers an inspection with adb, listens on the
bridge like the website, and records: detection time, Llama advice time and source, verdict, and
detections vs ground truth (same class, IoU >= 0.5). Then tests golden-board comparison on the phone:
inspect a reference board of the same design, send "golden_set" (as the website would), inspect the
test board again, and compare false alarms / found defects with and without the reference.

Sets:  PKU boards 10-12 (edited defects), PCB-Defect and DefectDet held-out real boards (7 classes),
       real clean phone frames (every box is a false alarm).
Usage: python3 ml/e2e_v3_phone.py   (bridge running, `adb reverse tcp:8765 tcp:8765`, app open)
Output: ml/results/phone/e2e_v3.json
"""
import asyncio
import json
import subprocess
import time
from pathlib import Path

import websockets

ML = Path(__file__).resolve().parent
ADB = str(Path.home() / "Library/Android/sdk/platform-tools/adb")
NAMES = ["missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper", "missing_pad"]
BOARDS = ["10_short_01", "11_open_circuit_01", "12_spur_01",
          "pcbd_50-1-1", "pcbd_50-1-2", "ddet_img102", "ddet_img18", "real_00", "real_01"]
GOLDEN = [("ddet_img102", "ddet_img126"),       # 100_1_BOTTOM_A checked against 100_1_BOTTOM_B (another physical board)
          ("10_short_01", "10_short_02")]       # PKU board 10, reference = another photo of the same board


def label_file(stem):
    if stem.startswith("pcbd_"):
        return ML / "data/real_test/pcbdefect/labels" / f"{stem[5:]}.txt"
    if stem.startswith("ddet_"):
        return ML / "data/real_test/defectdet/labels" / f"{stem[5:]}.txt"
    if stem.startswith("real_"):
        return None
    return ML / "data/pku/labels" / f"{stem}.txt"


def labels(stem, w, h):
    f = label_file(stem)
    out = []
    if f is None:
        return out
    for line in f.read_text().split("\n"):
        if line.strip():
            c, cx, cy, bw, bh = map(float, line.split()[:5])
            out.append((NAMES[int(c)], (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
    return out


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def score(stem, det):
    gt = labels(stem, det["width"], det["height"])
    used, tp, fp = set(), 0, 0
    for it in sorted(det["items"], key=lambda d: -d["conf"]):
        box = (it["x"], it["y"], it["x"] + it["w"], it["y"] + it["h"])
        best, bj = 0, None
        for j, g in enumerate(gt):
            if j not in used and g[0] == it["type"] and iou(box, g[1:]) > best:
                best, bj = iou(box, g[1:]), j
        if best >= 0.5:
            used.add(bj); tp += 1
        else:
            fp += 1
    return dict(true_defects=len(gt), found=tp, missed=len(gt) - tp, false_alarms=fp)


async def inspect(ws, stem):
    subprocess.run([ADB, "shell", "am", "start", "-n", "com.boardcheck.app/.MainActivity",
                    "--es", "image", f"files/test/{stem}.jpg"], capture_output=True)
    t0 = time.time(); det = t_det = None; advice = ""; status = {}
    while time.time() - t0 < 180:
        try:
            msg = await asyncio.wait_for(ws.recv(), timeout=5)
        except asyncio.TimeoutError:
            continue
        if isinstance(msg, bytes):
            continue
        m = json.loads(msg)
        if m.get("type") == "status":
            status = m
        elif m.get("type") == "not_pcb" and m.get("board_id") == stem:
            return dict(board=stem, not_pcb=m.get("pcb_prob"))
        elif m.get("type") == "detections" and m.get("board_id") == stem:
            det, t_det = m, time.time()
        elif m.get("type") == "advice_token" and det is not None:
            advice += m["text"]
        elif m.get("type") == "advice_done" and det is not None:
            r = dict(board=stem, **score(stem, det), verdict=m.get("verdict"),
                     items=[f"{i['type']} {i['conf']:.2f}" for i in det["items"]],
                     golden=det.get("golden"), golden_hidden=det.get("golden_hidden"),
                     phone_detect_ms=m.get("detect_ms"), phone_total_ms=m.get("ms"), tiles=det.get("tiles"),
                     ms_per_tile=det.get("ms_per_tile"), advice_source=m.get("source"),
                     llama_tok_s=status.get("llm_tps"), temp_c=status.get("temp"), thermal=status.get("thermal"),
                     quality=det.get("quality"), advice=(m.get("advice") or {}).get("summary", advice.strip()))
            return r
    return dict(board=stem, error="no result within 180 s")


async def command(ws, cmd):
    await ws.send(json.dumps({"type": "command", "cmd": cmd}))
    t0 = time.time()
    while time.time() - t0 < 30:
        try:
            m = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
        except (asyncio.TimeoutError, ValueError, TypeError):
            continue
        if isinstance(m, dict) and m.get("type") == "golden":
            return m.get("set")
    return None


async def run():
    out = {"boards": [], "golden": []}
    async with websockets.connect("ws://localhost:8765/ui", max_size=16 * 1024 * 1024) as ws:
        await command(ws, "golden_clear")
        for stem in BOARDS:
            r = await inspect(ws, stem); out["boards"].append(r)
            print(json.dumps({k: r.get(k) for k in ("board", "true_defects", "found", "missed", "false_alarms", "verdict",
                                                     "phone_detect_ms", "phone_total_ms", "ms_per_tile", "advice_source",
                                                     "llama_tok_s", "temp_c", "error", "not_pcb")}), flush=True)
        for test, ref in GOLDEN:
            await inspect(ws, ref)
            ok = await command(ws, "golden_set")
            r = await inspect(ws, test)
            await command(ws, "golden_clear")
            base = next((b for b in out["boards"] if b["board"] == test), {})
            g = dict(test=test, reference=ref, reference_saved=ok,
                     without=dict(found=base.get("found"), false_alarms=base.get("false_alarms")),
                     with_golden=dict(found=r.get("found"), false_alarms=r.get("false_alarms"), hidden=r.get("golden_hidden"),
                                      detect_ms=r.get("phone_detect_ms")))
            out["golden"].append(g); print("golden", json.dumps(g), flush=True)
    (ML / "results/phone").mkdir(parents=True, exist_ok=True)
    (ML / "results/phone/e2e_v3.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    asyncio.run(run())
