"""Run images already on the phone (files/test/<stem>.jpg) through the full app pipeline and record
what came back on the website link: verdict, detections, advice source, timings.
Usage: python3 run_phone_images.py out.json stem1 stem2 ...   (bridge + app must be running, USB link on)"""
import asyncio, json, subprocess, sys, time
from pathlib import Path
import websockets
ADB = str(Path.home() / "Library/Android/sdk/platform-tools/adb")

async def run(out, stems):
    res = []
    async with websockets.connect("ws://localhost:8765/ui", max_size=16 * 1024 * 1024) as ws:
        for stem in stems:
            subprocess.run([ADB, "shell", "am", "start", "-n", "com.boardcheck.app/.MainActivity", "--es", "image", f"files/test/{stem}.jpg"], capture_output=True)
            t0 = time.time(); r = dict(board=stem)
            while time.time() - t0 < 60:
                try: m = await asyncio.wait_for(ws.recv(), timeout=5)
                except asyncio.TimeoutError: continue
                if isinstance(m, bytes): continue
                m = json.loads(m); t = m.get("type")
                if m.get("board_id") not in (None, stem) and t in ("detections", "not_pcb"): continue
                if t == "not_pcb": r.update(verdict="not_pcb", pcb_prob=m.get("pcb_prob"), items=[]); break
                if t == "detections": r.update(items=[(i["type"], round(i["conf"], 2)) for i in m["items"]], quality=m.get("quality"), detect_ms=m.get("detect_ms"))
                if t == "advice_done" and "items" in r:
                    r.update(verdict=m["verdict"], total_ms=m["ms"], source=m.get("source"), grounded=(m.get("advice") or {}).get("grounded"),
                             summary=(m.get("advice") or {}).get("summary")); break
            print(json.dumps(r)); res.append(r)
    Path(out).write_text(json.dumps(res, indent=1))

asyncio.run(run(sys.argv[1], sys.argv[2:]))
