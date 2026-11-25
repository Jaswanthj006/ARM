"""Camera-in-the-loop test: the laptop shows held-out test boards full-screen (http://localhost:8000/screen.html),
the phone photographs the screen with its real camera and runs the full app pipeline.
Box positions cannot be matched to labels (camera geometry), so scoring is per board:
defects found per type (count matched, capped at the true count), extra detections, and verdict.
Usage: python3 ml/screen_camera_test.py [n_boards]   (bridge running, app linked, phone aimed at the screen)"""
import asyncio, json, shutil, sys, time
from collections import Counter
from pathlib import Path
import websockets
ROOT = Path(__file__).resolve().parent; WEB = ROOT.parent / "web/screen"
NAMES = ["missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"]
RANK = {"short": 3, "open_circuit": 3, "missing_hole": 2, "mouse_bite": 2, "spur": 1, "spurious_copper": 1}
def truth(stem): return [NAMES[int(l.split()[0])] for l in (ROOT / "data/pku/labels" / f"{stem}.txt").read_text().split("\n") if l.strip()]
def verdict(types): return "pass" if not types else ("reject" if max(RANK[t] for t in types) == 3 else "rework")

async def run(n):
    stems = sorted(p.stem for p in (ROOT / "data/pku/images").glob("1[0-2]_*.jpg"))
    stems = stems[:: max(1, len(stems) // n)][:n]
    res = []
    async with websockets.connect("ws://localhost:8765/ui", max_size=16 * 1024 * 1024) as ws:
        for k, stem in enumerate(stems):
            shutil.copy(ROOT / "data/pku/images" / f"{stem}.jpg", WEB / "board.jpg")
            (WEB / "current.json").write_text(json.dumps({"img": f"board.jpg#{stem}", "label": f"{k + 1}/{len(stems)} {stem}"}))
            await asyncio.sleep(2.0)                                     # screen refresh + camera autofocus
            await ws.send(json.dumps({"cmd": "capture", "board_id": stem}))
            t0 = time.time(); r = dict(board=stem)
            while time.time() - t0 < 60:
                try: m = await asyncio.wait_for(ws.recv(), timeout=5)
                except asyncio.TimeoutError: continue
                if isinstance(m, bytes): continue
                m = json.loads(m); t = m.get("type")
                if t == "not_pcb": r.update(pred=[], verdict="not_pcb", pcb_prob=m.get("pcb_prob")); break
                if t == "detections": r.update(pred=[i["type"] for i in m["items"]], quality=m.get("quality"), detect_ms=m.get("detect_ms"))
                if t == "advice_done" and "pred" in r:
                    r.update(verdict=m["verdict"], total_ms=m["ms"], source=m.get("source"), grounded=(m.get("advice") or {}).get("grounded")); break
            if "verdict" not in r: r["error"] = "timeout"; print(stem, "TIMEOUT"); res.append(r); continue
            gt, pr = Counter(truth(stem)), Counter(r["pred"])
            found = sum(min(gt[t], pr[t]) for t in gt)
            r.update(true=sum(gt.values()), found=found, extra=sum(pr.values()) - sum(min(gt[t], pr[t]) for t in pr),
                     true_verdict=verdict(list(gt)), verdict_ok=r["verdict"] == verdict(list(gt)))
            print(f"{stem:24} true {r['true']} found {found} extra {r['extra']} {r['verdict']:8} ok={r['verdict_ok']} "
                  f"detect {r.get('detect_ms')} ms total {r.get('total_ms')} ms q={r.get('quality')}")
            res.append(r)
    ok = [r for r in res if "error" not in r]
    T = sum(r["true"] for r in ok); F = sum(r["found"] for r in ok); E = sum(r["extra"] for r in ok)
    summ = dict(boards=len(res), timeouts=len(res) - len(ok), true_defects=T, found=F, recall=round(F / max(1, T), 3),
                extra_detections=E, precision=round(F / max(1, F + E), 3),
                verdict_accuracy=round(sum(r["verdict_ok"] for r in ok) / max(1, len(ok)), 3),
                not_pcb=sum(r["verdict"] == "not_pcb" for r in ok), quality_warnings=sum(bool(r.get("quality")) for r in ok),
                median_detect_ms=sorted(r.get("detect_ms") or 0 for r in ok)[len(ok) // 2] if ok else None,
                median_total_ms=sorted(r.get("total_ms") or 0 for r in ok)[len(ok) // 2] if ok else None)
    print("SUMMARY", json.dumps(summ))
    (ROOT / "results/phone/screen_camera_test.json").write_text(json.dumps(dict(summary=summ, runs=res), indent=1))
    (WEB / "current.json").write_text(json.dumps({"img": "", "label": "done"}))

asyncio.run(run(int(sys.argv[1]) if len(sys.argv) > 1 else 20))
