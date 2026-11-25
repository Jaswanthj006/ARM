#!/bin/bash
# Build and install the BoardCheck app.
# The Kotlin compiler cannot handle "!" in paths (this repo lives in "ARM!!"),
# so the sources are synced to ~/boardcheck-build and built there.
set -e
SRC="$(cd "$(dirname "$0")" && pwd)"
DST="$HOME/boardcheck-build"
mkdir -p "$DST"
rsync -a --delete --exclude build --exclude .gradle --exclude .kotlin --exclude local.properties "$SRC/" "$DST/"
echo "sdk.dir=$HOME/Library/Android/sdk" > "$DST/local.properties"
export JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home"
cd "$DST"
./gradlew assembleDebug --console=plain -q
APK="$DST/app/build/outputs/apk/debug/app-debug.apk"
cp "$APK" "$SRC/BoardCheck-debug.apk"
echo "APK: $SRC/BoardCheck-debug.apk ($(du -h "$APK" | cut -f1))"
if [ "$1" = "install" ]; then
  "$HOME/Library/Android/sdk/platform-tools/adb" install -r "$APK"
fi
