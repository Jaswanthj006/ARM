"""End-to-end test of the real system: phone app -> bridge -> (this script as the website).

For each test board image stored on the phone, triggers an inspection with adb,
listens on the bridge like the website does, and records:
  * time to detections (PCB check + tiling + YOLO on the phone)
  * time to finished advice (Llama on the phone)
  * detections vs ground-truth labels (IoU >= 0.5): found, missed, false alarms
  * the advice text
Results go to ml/results/phone/e2e.json.

Usage:  python3 ml/e2e_phone_test.py   (bridge running, phone connected, app open)
"""
import asyncio
import json
import subprocess
import time
from pathlib import Path

import websockets

ROOT = Path(__file__).resolve().parent
ADB = str(Path.home() / "Library/Android/sdk/platform-tools/adb")
BOARDS = ["10_short_01", "10_short_02", "11_open_circuit_01", "12_missing_hole_01", "12_spur_01"]
NAMES = ["missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"]


def labels(stem, w, h):
    out = []
    for line in (ROOT / "data/pku/labels" / f"{stem}.txt").read_text().split("\n"):
        if line.strip():
            c, cx, cy, bw, bh = map(float, line.split())
            out.append((NAMES[int(c)], (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
    return out


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


async def run():
    results = []
    async with websockets.connect("ws://localhost:8765/ui", max_size=16 * 1024 * 1024) as ws:
        for stem in BOARDS:
            subprocess.run([ADB, "shell", "am", "start", "-n", "com.boardcheck.app/.MainActivity",
                            "--es", "image", f"files/test/{stem}.jpg"], capture_output=True)
            t0 = time.time(); src = phone_detect_ms = None; det = None; t_det = t_done = None; advice = ""; verdict = None; status = {}
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
                    det = {"items": [], "width": 1, "height": 1, "not_pcb": m.get("pcb_prob")}; t_det = t_done = time.time()
                    break
                elif m.get("type") == "detections" and m.get("board_id") == stem:
                    det, t_det = m, time.time()
                elif m.get("type") == "advice_token" and det is not None:
                    advice += m["text"]
                elif m.get("type") == "advice_done" and det is not None:
                    verdict, t_done = m.get("verdict"), time.time(); src = m.get("source"); phone_detect_ms = m.get("detect_ms")
                    break
            if det is None:
                results.append(dict(board=stem, error="no result within 120 s")); continue
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
            r = dict(board=stem, true_defects=len(gt), found=tp, missed=len(gt) - len(used), false_alarms=fp,
                     verdict=verdict, pcb_check="failed" if det.get("not_pcb") is not None else "passed",
                     detect_s=round(t_det - t0, 1), total_s=round(t_done - t0, 1) if t_done else None, advice_finished=t_done is not None,
                     yolo_ms_per_tile=status.get("latency_ms"), llama_tok_s=status.get("llm_tps"),
                     phone_temp_c=status.get("temp"), thermal=status.get("thermal"), advice_source=src, phone_detect_ms=phone_detect_ms, advice=advice.strip())
            results.append(r)
            print(f"{stem:22} true {r['true_defects']}  found {tp}  missed {r['missed']}  false {fp}  "
                  f"{verdict}  detect {r['detect_s']}s  total {r['total_s']}s  llama {r['llama_tok_s']} tok/s  {r['phone_temp_c']}°C")
            await asyncio.sleep(3)
    ok = [r for r in results if "error" not in r]
    summary = dict(boards=len(results), true_defects=sum(r["true_defects"] for r in ok), found=sum(r["found"] for r in ok),
                   missed=sum(r["missed"] for r in ok), false_alarms=sum(r["false_alarms"] for r in ok),
                   median_detect_s=sorted(r["detect_s"] for r in ok)[len(ok) // 2] if ok else None,
                   median_total_s=sorted(r["total_s"] for r in ok if r["total_s"])[len([r for r in ok if r['total_s']]) // 2] if any(r['total_s'] for r in ok) else None)
    out = ROOT / "results/phone/e2e.json"; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(summary=summary, runs=results), indent=2))
    print("\nSUMMARY", json.dumps(summary))


if __name__ == "__main__":
    asyncio.run(run())
