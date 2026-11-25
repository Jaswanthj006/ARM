#!/bin/bash
# A/B test of big-core pinning: same board, Llama cache off, alternating off/on so heat affects both equally.
A=~/Library/Android/sdk/platform-tools/adb
IMG=${1:-ddet_img18}; ROUNDS=${2:-3}
$A shell am force-stop com.boardcheck.app; $A shell am start -n com.boardcheck.app/.MainActivity >/dev/null; sleep 20
for r in $(seq 1 $ROUNDS); do for pin in false true; do
  $A logcat -c
  $A shell am start -n com.boardcheck.app/.MainActivity --es image files/test/$IMG.jpg --ez no_cache true --ez pin_big $pin >/dev/null 2>&1
  for i in $(seq 1 90); do sleep 1; $A logcat -d -s BoardCheck | grep -q "status: .*total" && break; done
  y=$($A logcat -d -s BoardCheck | grep "yolo profile" | sed 's/.*fwd=\([0-9]*\).*/\1/')
  l=$($A logcat -d -s BoardCheck | grep "llama done" | sed 's/.*prompt \([0-9]*\) tok in \([0-9]*\) ms.*decode \([0-9]*\) tok in \([0-9]*\) ms.*/prefill_ms=\2 decode_tok=\3 decode_ms=\4/')
  t=$($A logcat -d -s BoardCheck | grep "status: .*total" | sed 's/.*total \([0-9.]*\) s.*/\1/')
  th=$($A shell dumpsys thermalservice | grep 'Thermal Status' | tr -dc 0-9); b=$($A shell dumpsys battery | grep temperature | tr -dc 0-9)
  echo "round=$r pin=$pin yolo_fwd_ms_per_tile=$y $l total_s=$t thermal=$th battery=$b"
done; done
