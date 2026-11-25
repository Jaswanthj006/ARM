#!/bin/bash
# Builds libbcannotate.so (arm64) into app/src/main/jniLibs. TCP mode: Android 11+ blocks unix sockets between
# the app user and gatord's user.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
NDK="$HOME/Library/Android/sdk/ndk/28.2.13676358"
CC="$NDK/toolchains/llvm/prebuilt/darwin-x86_64/bin/aarch64-linux-android28-clang"
OUT="$HERE/../../app/src/main/jniLibs/arm64-v8a"; mkdir -p "$OUT"
"$CC" -O2 -fPIC -shared -DTCP_ANNOTATIONS -o "$OUT/libbcannotate.so" "$HERE/annotate_jni.c" "$HERE/affinity_jni.c" "$HERE/streamline_annotate.c" -llog
echo "built $OUT/libbcannotate.so"
