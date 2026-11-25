"""Draws the inspection timeline from a Streamline capture's annotation log (profiling/captures/<name>_all/log.csv)
plus the share of cycles on the big A715 cores per stage (analyze.py summary JSON).
usage: python profiling/plot_timeline.py <log.csv> <summary.json> <out.png>"""
import csv, json, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

log, summ, out = sys.argv[1:4]
rows = [r for r in csv.reader(open(log)) if len(r) > 4 and r[1] and r[0][0].isdigit()]
t0 = min(float(r[0]) for r in rows)
big = {s["stage"]: s["big_core_share"] for s in json.load(open(summ))}
COL = {"Board check (gate)": "#7B5CC4", "Photo quality": "#B4B2A9", "YOLO11n defect detection": "#1D9E75",
       "NMS": "#5DCAA5", "Llama prefill (board findings)": "#EF9F27", "Llama decode": "#C77700"}
fig, ax = plt.subplots(figsize=(11, 3.1), dpi=200)
stages = [r for r in rows if r[4] == "Stage"]
for r in stages:
    s, d, name = float(r[0]) - t0, float(r[1]), r[2]
    ax.barh(1, d, left=s, color=COL.get(name, "#888"), edgecolor="white", height=0.55)
    if d > 0.4:
        b = big.get(name)
        ax.text(s + d / 2, 1, f"{name.replace(' (board findings)', '').replace(' (gate)', '')}\n{d:.1f} s" + (f" · {b*100:.0f}% on A715" if b else ""),
                ha="center", va="center", fontsize=7.5, color="white", fontweight="bold")
tiles = [r for r in rows if r[4] == "YOLO tile"]
for k, r in enumerate(tiles):
    ax.barh(0.25, float(r[1]), left=float(r[0]) - t0, color="#0F6E56" if k % 2 else "#5DCAA5", height=0.3)
ax.text(float(tiles[0][0]) - t0, -0.05, f"{len(tiles)} YOLO tiles, {sum(float(r[1]) for r in tiles)/len(tiles)*1000:.0f} ms each", fontsize=7, va="top")
small = [r for r in stages if float(r[1]) <= 0.4]
ax.text(0, 1.45, "Board check " + " · ".join(f"{float(r[1])*1000:.0f} ms" for r in small[:1]) + "  ·  photo quality, NMS < 0.1 s", fontsize=7, color="#5F5E5A")
ax.set_yticks([]); ax.set_xlabel("seconds from start of inspection (phone at Android thermal status 3, throttled)", fontsize=8)
ax.set_ylim(-0.35, 1.6); ax.set_xlim(0, max(float(r[0]) + float(r[1]) for r in stages) - t0 + 0.1)
for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
ax.set_title("One inspection on the phone, from the Arm Streamline capture (stage labels written by the app)", fontsize=9, loc="left")
plt.tight_layout(); plt.savefig(out, facecolor="white"); print("saved", out)
