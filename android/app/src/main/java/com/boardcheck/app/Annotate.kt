package com.boardcheck.app

/**
 * Stage labels for Arm Streamline (Arm Performance Studio). Each stage of an inspection shows as a named,
 * coloured block in Streamline's timeline, so CPU activity can be matched to "board check", "YOLO tile 3/12",
 * "Llama prefill" and so on. Uses Arm's annotation library through libbcannotate.so (android/native/annotate).
 * When no Streamline capture is running, or the library is missing, every call does nothing.
 */
object Annotate {
    const val CH_INSPECT = 1; const val CH_STAGE = 2; const val CH_TILE = 3
    // Colours in Streamline's 0xBBGGRR1b format (see streamline_annotate.h).
    const val RED = 0x0000ff1b; const val BLUE = 0xff00001b.toInt(); const val GREEN = 0x00ff001b
    const val PURPLE = 0xff00ff1b.toInt(); const val YELLOW = 0x00ffff1b; const val CYAN = 0xffff001b.toInt()

    private val ok = runCatching { System.loadLibrary("bcannotate"); nSetup()
        nName(CH_INSPECT, "Inspection"); nName(CH_STAGE, "Stage"); nName(CH_TILE, "YOLO tile") }.isSuccess

    fun begin(ch: Int, color: Int, name: String) { if (ok) nBegin(ch, color, name) }
    fun end(ch: Int) { if (ok) nEnd(ch) }
    fun marker(color: Int, name: String) { if (ok) nMarker(color, name) }
    inline fun <T> span(ch: Int, color: Int, name: String, block: () -> T): T { begin(ch, color, name); try { return block() } finally { end(ch) } }

    @JvmStatic private external fun nSetup()
    @JvmStatic private external fun nName(ch: Int, name: String)
    @JvmStatic private external fun nBegin(ch: Int, color: Int, name: String)
    @JvmStatic private external fun nEnd(ch: Int)
    @JvmStatic private external fun nMarker(color: Int, name: String)
}
