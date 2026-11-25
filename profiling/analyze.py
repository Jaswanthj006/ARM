"""Summarise an Arm Streamline capture of BoardCheck, per inspection stage.

Input: the per-thread CSV export of a capture (Streamline -report ... -all -individual-threads -process com.boardcheck.app)
and the annotation log (the stage labels the app writes through Annotate.kt).

For every stage (board check, YOLO detection, Llama prefill, Llama decode) it reports, from the CPU's own counters:
  - time, and how many CPU cycles ran on the big Cortex-A715 cores vs the small Cortex-A510 cores
  - IPC (instructions per cycle) on each core type
  - L2 cache refills per 1000 instructions, and back-end stall share (A715)
  - which threads did the work, and how much CPU the camera / UI / barcode threads took meanwhile

usage: python3 profiling/analyze.py <report_dir> [out.json]
"""
import csv, glob, json, os, re, sys
from collections import defaultdict

rep = sys.argv[1]
out_json = sys.argv[2] if len(sys.argv) > 2 else None

def rows(path, start_marker):
    r = list(csv.reader(open(path)))
    for i, x in enumerate(r):
        if x and x[0].startswith(start_marker):
            return x, r[i + 1:]
    return None, []

# Stage intervals from the annotation log.
_, log = rows(os.path.join(rep, "log.csv"), "When (s)")
stages = [(float(x[0]), float(x[1]), x[2], x[4]) for x in log if len(x) > 4 and x[1]]

# Per-thread counter timelines (1 ms bins).
NAME = re.compile(r"for thread '(.*)', tid: (\d+)")
threads = {}
for f in glob.glob(os.path.join(rep, "timeline_for_thread_*.csv")):
    head = open(f).read(400)
    m = NAME.search(head)
    hdr, data = rows(f, "Index (s)")
    if not hdr or not data:
        continue
    col = {h: i for i, h in enumerate(hdr)}
    t = [float(x[0]) for x in data]
    def series(key):
        i = col.get(key)
        return [float(x[i].replace(",", "")) if i is not None and x[i] not in ("", "-") else 0.0 for x in data]
    threads[f"{m.group(1)}#{m.group(2)}"] = dict(t=t,
        c715=series("Cycles (Cortex-A715):CPU Cycles"), c510=series("Cycles (Cortex-A510):CPU Cycles"),
        i715=series("Instructions (Executed) (Cortex-A715):All"), i510=series("Instructions (Executed) (Cortex-A510):All"),
        l2_715=series("L2 Data Cache (Cortex-A715):Refill"), l2_510=series("L2 Data Cache (Cortex-A510):Refill"),
        be=series("Stalls (Cortex-A715):Backend"), fe=series("Stalls (Cortex-A715):Frontend"))

def window(th, a, b):
    t = th["t"]; lo = next((k for k, v in enumerate(t) if v >= a), len(t)); hi = next((k for k, v in enumerate(t) if v > b), len(t))
    return {k: sum(v[lo:hi]) for k, v in th.items() if k != "t"}

AI = re.compile(r"pool-4-thread|^com\.boardcheck\.app#\d+$")   # inspection thread + unnamed native (ExecuTorch / XNNPACK) workers
SIDE = re.compile(r"CameraX|RenderThread|hwui|MLHandler|pool-8|pool-5|pool-7|barcode|GoogleApi", re.I)

summary = []
for (s, d, name, ch) in stages:
    if ch == "YOLO tile":
        continue
    tot = defaultdict(float); per = []; side = 0.0
    for tn, th in threads.items():
        w = window(th, s, s + d)
        cyc = w["c715"] + w["c510"]
        if cyc <= 0:
            continue
        if SIDE.search(tn):
            side += cyc
        for k, v in w.items():
            tot[k] += v
        per.append((tn, w["c715"], w["c510"]))
    per.sort(key=lambda x: -(x[1] + x[2]))
    c = tot["c715"] + tot["c510"]
    row = dict(stage=name, ms=round(d * 1000),
               big_core_share=round(tot["c715"] / c, 3) if c else None,
               ipc_a715=round(tot["i715"] / tot["c715"], 2) if tot["c715"] else None,
               ipc_a510=round(tot["i510"] / tot["c510"], 2) if tot["c510"] else None,
               l2_refill_per_1k_instr_a715=round(1000 * tot["l2_715"] / tot["i715"], 2) if tot["i715"] else None,
               backend_stall_a715=round(tot["be"] / tot["c715"], 3) if tot["c715"] else None,
               frontend_stall_a715=round(tot["fe"] / tot["c715"], 3) if tot["c715"] else None,
               avg_busy_cores=round(c / (d * 1e9) / 2.4, 2) if d else None,   # rough: cycles / (time x ~2.4 GHz average clock)
               side_threads_share=round(side / c, 3) if c else None,
               top_threads=[dict(thread=t, a715_Mcyc=round(a / 1e6, 1), a510_Mcyc=round(b / 1e6, 1)) for t, a, b in per[:6]])
    summary.append(row)

for r in summary:
    print(f"\n== {r['stage']}  ({r['ms']} ms)")
    for k in ("big_core_share", "ipc_a715", "ipc_a510", "l2_refill_per_1k_instr_a715", "backend_stall_a715", "frontend_stall_a715", "side_threads_share"):
        print(f"   {k:32s} {r[k]}")
    for t in r["top_threads"]:
        print(f"   {t['thread'][:34]:34s} A715 {t['a715_Mcyc']:8.1f} Mcyc   A510 {t['a510_Mcyc']:8.1f} Mcyc")
if out_json:
    json.dump(summary, open(out_json, "w"), indent=1)
