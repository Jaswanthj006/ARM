// JNI bridge: lets the Kotlin app label its stages (board check, YOLO tiles, Llama) in Arm Streamline's timeline.
// Uses Arm's annotation library (streamline_annotate.c, from Arm Performance Studio). Without a Streamline
// capture running, every call is a cheap no-op.
#include <jni.h>
#include "streamline_annotate.h"

JNIEXPORT void JNICALL Java_com_boardcheck_app_Annotate_nSetup(JNIEnv *e, jclass c) { gator_annotate_setup(); }

JNIEXPORT void JNICALL Java_com_boardcheck_app_Annotate_nName(JNIEnv *e, jclass c, jint ch, jstring s) {
  const char *t = (*e)->GetStringUTFChars(e, s, 0); gator_annotate_name_channel((uint32_t) ch, 0, t); (*e)->ReleaseStringUTFChars(e, s, t);
}
JNIEXPORT void JNICALL Java_com_boardcheck_app_Annotate_nBegin(JNIEnv *e, jclass c, jint ch, jint color, jstring s) {
  const char *t = (*e)->GetStringUTFChars(e, s, 0); gator_annotate_color((uint32_t) ch, (uint32_t) color, t); (*e)->ReleaseStringUTFChars(e, s, t);
}
JNIEXPORT void JNICALL Java_com_boardcheck_app_Annotate_nEnd(JNIEnv *e, jclass c, jint ch) { gator_annotate_str((uint32_t) ch, 0); }

JNIEXPORT void JNICALL Java_com_boardcheck_app_Annotate_nMarker(JNIEnv *e, jclass c, jint color, jstring s) {
  const char *t = (*e)->GetStringUTFChars(e, s, 0); gator_annotate_marker_color((uint32_t) color, t); (*e)->ReleaseStringUTFChars(e, s, t);
}
