#!/bin/bash
# Arm Streamline capture of BoardCheck on the phone (no root needed: the debug app is profileable).
# usage: profiling/capture.sh <name> <seconds> <image1> [image2 ...]
# gatord launches the app; after the models load, each test image is inspected (Llama cache off), one after another.
set -e
NAME=$1; DUR=$2; shift 2
A=~/Library/Android/sdk/platform-tools/adb
HERE="$(cd "$(dirname "$0")" && pwd)"; OUT="$HERE/captures"; mkdir -p "$OUT"
$A push "$HERE/configuration.xml" /data/local/tmp/configuration.xml >/dev/null
$A shell rm -rf /data/local/tmp/$NAME.apc
$A shell am force-stop com.boardcheck.app
echo "start: battery $($A shell dumpsys battery | grep temperature | tr -dc 0-9)/10 C, thermal $($A shell dumpsys thermalservice | grep 'Thermal Status' | tr -dc 0-9)"
$A shell "cd /data/local/tmp && ./gatord --android-pkg com.boardcheck.app --android-activity com.boardcheck.app.MainActivity --stop-on-exit yes --max-duration $DUR --output /data/local/tmp/$NAME.apc --gpu-timeline no > /data/local/tmp/$NAME.log 2>&1" &
GP=$!
sleep 20                                   # app start + model loading (Llama ~1 GB)
$A logcat -c
for img in "$@"; do
  $A shell am start -n com.boardcheck.app/.MainActivity --es image files/test/$img.jpg --ez no_cache true >/dev/null 2>&1
  for i in $(seq 1 60); do sleep 1; $A logcat -d -s BoardCheck | grep -q "status: .*total" && break; done
  $A logcat -d -s BoardCheck | grep -E "status:|llama done|yolo profile" | cut -c34-230; $A logcat -c
done
wait $GP || true
echo "end: battery $($A shell dumpsys battery | grep temperature | tr -dc 0-9)/10 C, thermal $($A shell dumpsys thermalservice | grep 'Thermal Status' | tr -dc 0-9)"
rm -rf "$OUT/$NAME.apc"; $A pull /data/local/tmp/$NAME.apc "$OUT/" >/dev/null
(cd "$OUT" && rm -f $NAME.apc.zip && zip -qr $NAME.apc.zip $NAME.apc)
du -sh "$OUT/$NAME.apc.zip"
