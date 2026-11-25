package com.boardcheck.app

/**
 * Big-core pinning for an inspection (android/native/annotate/affinity_jni.c).
 * Arm Streamline showed 27-42 % of inference cycles on the small Cortex-A510 cores; pinning keeps
 * YOLO and Llama on the Cortex-A715 cores while a board is inspected, then frees all cores again.
 */
object Affinity {
    private val ok = runCatching { System.loadLibrary("bcannotate") }.isSuccess
    /** Returns how many cores are allowed (0 = all / not available). */
    fun pinBig(on: Boolean): Int = if (ok) runCatching { nPin(on) }.getOrDefault(0) else 0
    @JvmStatic private external fun nPin(big: Boolean): Int
}
