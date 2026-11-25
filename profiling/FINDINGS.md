# BoardCheck on Arm: Streamline profile

Tool: Arm Performance Studio 2026.6, Streamline 9.9.1 (gatord, no root; debug app).
Phone: Nothing Phone 2a (MediaTek MT6886), 2 x Cortex-A715 @ 2.8 GHz (cpu6-7) + 6 x Cortex-A510 @ 2.0 GHz, Android 16.
Workload: one full inspection of `ddet_img18` (1703 x 1920 photo, 16 YOLO tiles, Llama advice with cache off).

## How it was captured

- Stage labels: the app writes Streamline annotations through Arm's annotation library
  (`android/native/annotate`, `Annotate.kt`). The timeline shows "Inspect <board>", "Board check (gate)",
  "Photo quality", "YOLO11n defect detection" with every "YOLO tile i/n", "NMS", "Llama prefill", "Llama decode".
- Counters (`profiling/configuration.xml`): cycles, instructions, L1D/L2D refills, branch mispredicts,
  front-end/back-end stalls on the A715 cores; cycles, instructions, L1D/L2D refills on the A510 cores.
- Scripts: `profiling/capture.sh` (capture), `profiling/analyze.py` (per-stage summary from the per-thread CSV export),
  `profiling/ab_pin.sh` (pinned vs unpinned A/B).

## Capture 1: one inspection (phone warm: battery 42-43 C, Android thermal status 3)

| Stage | Time | Cycles on A715 | IPC A715 | IPC A510 | L2 refills / 1k instr (A715) | Back-end stalls (A715) |
|---|---|---|---|---|---|---|
| Board check (gate) | 133 ms | 72 % | 2.94 | 0.65 | 1.8 | 21 % |
| Photo quality | 91 ms | 79 % | 2.75 | 0.74 | 0.6 | 37 % |
| YOLO11n, 16 tiles | 3.25 s (203 ms/tile) | 73 % | 2.42 | 0.89 | 2.9 | 22 % |
| Llama prefill (158 tokens) | 4.79 s | 73 % | 2.35 | 0.74 | 2.3 | 25 % |
| Llama decode (48 tokens) | 6.75 s | 58 % | 2.27 | 0.70 | 8.6 | 42 % |

What it shows:
1. **Two threads do all the AI work** (the inspection thread and one ExecuTorch/XNNPACK worker), as configured
   (2 threads). Camera, UI and barcode threads take only 1-3 % of the cycles during an inspection.
2. **27-42 % of the inference cycles run on the small A510 cores**, which retire about 3x fewer instructions per
   cycle (IPC 0.7-0.9 vs 2.3-2.9) at a lower clock. The scheduler moves the threads there, most during Llama decode.
3. **Llama decode is memory-bound**: 42 % back-end stalls and 8.6 L2 refills per 1000 instructions (3-4x prefill),
   as expected for token-by-token decoding that streams the 1B model weights. YOLO and prefill are compute-bound
   (IPC 2.3-2.4, few refills), where the XNNPACK FP16 / int8 kernels keep the A715 busy.
4. **Heat is the biggest factor on this phone**: on USB charge the phone sat at thermal status 2-3; the big cores
   were capped at 1.5 GHz (of 2.8) and the small cores at 1.05 GHz (of 2.0). YOLO went from ~130 ms/tile cool to
   200-290 ms/tile hot.

## Optimisation tried: keep inference on the big cores

`Affinity.kt` + `affinity_jni.c`: during an inspection every thread of the app may only run on the fastest cores
(cpu6-7, found from `cpuinfo_max_freq`); afterwards all cores are allowed again. Verified on the phone:
`Cpus_allowed_list` = 6-7 during the inspection, 0-7 after. Switch: adb extra `--ez pin_big true` (off by default).

A/B while thermal status was 1-3 (`ab_pin_run1.txt`): no clear result. Run-to-run spread from throttling
(YOLO 135-288 ms/tile) is larger than the effect. **To do:** repeat with a cool phone.
