package com.boardcheck.app

import android.os.SystemClock
import org.pytorch.executorch.extension.llm.LlmCallback
import org.pytorch.executorch.extension.llm.LlmModule
import org.pytorch.executorch.extension.llm.LlmModuleConfig

/**
 * Llama 3.2 1B advisor (ExecuTorch, 4-bit, Arm CPU).
 *
 * Gets structured findings, never pixels (PS5 design hint), and writes
 * SUMMARY / STEP / CAUSE lines. The pass / rework / reject decision is fixed by
 * rules in QcAdvisor; Llama explains it. Clean boards skip Llama.
 *
 * Speed architecture (needs the patched ExecuTorch in executorch_android/):
 *  - Prefix cache: the fixed system prompt is prefilled ONCE at load. Each board
 *    rewinds the KV cache to the end of it and prefills only the board's findings.
 *  - Threads: one ExecuTorch pool for all models, fixed at 2 (the two big A715
 *    cores). XNNPACK plans its work for the thread count a model was loaded with,
 *    so the count is never changed after loading (switching per phase broke YOLO).
 */
class Advisor(modelPath: String, tokenizerPath: String) {

    data class Stats(val promptTokens: Int, val generatedTokens: Int, val firstTokenMs: Long, val decodeTokPerS: Double, val totalMs: Long)

    companion object {
        const val MAX_TOKENS = 48
        const val TIMEOUT_MS = 25_000L
        const val THREADS = 2
    }

    private val llm: LlmModule
    private var prefixPos = -1L
    @Volatile private var timedOut = false

    init {
        // Read by the patched ExecuTorch JNI when it creates the shared thread pool.
        android.system.Os.setenv("EXECUTORCH_NUM_THREADS", THREADS.toString(), true)
        llm = LlmModule(
            LlmModuleConfig.create().modulePath(modelPath).tokenizerPath(tokenizerPath)
                .temperature(0.0f).loadMode(LlmModuleConfig.LOAD_MODE_MMAP).build()
        )
        llm.load()
        cachePrefix()
    }

    private fun cachePrefix() {
        val t = SystemClock.elapsedRealtime()
        llm.resetContext()
        llm.prefillPrompt(QcAdvisor.SYSTEM_PREFIX)
        prefixPos = llm.position()
        android.util.Log.i("BoardCheck", "llama prefix cached: $prefixPos tokens in ${SystemClock.elapsedRealtime() - t} ms")
    }

    // Advice cache: the same defect pattern (types, worst location, trend) gets the same advice instantly.
    private val cache = object : LinkedHashMap<String, QcAdvisor.Advice>(32, 0.75f, true) {
        override fun removeEldestEntry(e: MutableMap.MutableEntry<String, QcAdvisor.Advice>) = size > 50
    }

    /** Blocking; call from a worker thread. onToken gets each piece of text as it is generated. */
    /** false while profiling, so every board runs Llama (adb: --ez no_cache true). */
    var useCache = true

    fun advise(boardId: String, findings: List<QcAdvisor.Finding>, trend: String? = null, onToken: (String) -> Unit): Pair<QcAdvisor.Advice, Stats?> {
        if (findings.isEmpty()) return QcAdvisor.CLEAN to null
        val w = QcAdvisor.sortWorst(findings).first()
        val key = findings.map { it.type }.toSortedSet().joinToString(",") + "|" + w.type + "@" + w.location + "|" + trend
        if (useCache) cache[key]?.let { onToken(it.summary); return it.copy(source = "cache") to null }

        val reply = StringBuilder()
        var stopped = false
        var n = 0
        val t0 = SystemClock.elapsedRealtime()
        // Safety net: never let Llama hold the operator longer than TIMEOUT_MS.
        val watchdog = Thread { try { Thread.sleep(TIMEOUT_MS); if (!stopped) { stopped = true; timedOut = true; llm.stop() } } catch (_: InterruptedException) {} }
        timedOut = false
        try {
            if (prefixPos <= 0 || !llm.rewind(prefixPos)) Annotate.span(Annotate.CH_STAGE, Annotate.YELLOW, "Llama system prompt") { cachePrefix() }          // reuse the cached system prompt
            onToken(QcAdvisor.PREFILL + " ")
            watchdog.start()
            Annotate.span(Annotate.CH_STAGE, Annotate.PURPLE, "Llama prefill (board findings)") {
                llm.prefillPrompt(QcAdvisor.boardPart(boardId, findings, trend)) }   // only the new tokens
            val promptTokens = (llm.position() - prefixPos).toInt()
            val t1 = SystemClock.elapsedRealtime()
            Annotate.begin(Annotate.CH_STAGE, Annotate.BLUE, "Llama decode")
            if (!stopped) llm.generate("", 1024, object : LlmCallback {
                override fun onResult(result: String) {
                    if (stopped) return
                    if (result.contains("<|")) { stopped = true; llm.stop(); return }
                    reply.append(result); n++
                    onToken(result)
                    if (QcAdvisor.complete(reply.toString()) || n >= MAX_TOKENS) { stopped = true; llm.stop() }
                }
            }, false)
            Annotate.end(Annotate.CH_STAGE)
            stopped = true; watchdog.interrupt()
            val t2 = SystemClock.elapsedRealtime()
            val st = Stats(promptTokens, n, t1 - t0, if (t2 > t1) n * 1000.0 / (t2 - t1) else 0.0, t2 - t0)
            android.util.Log.i("BoardCheck", "llama done: board prompt $promptTokens tok in ${t1 - t0} ms " +
                "(${if (t1 > t0) promptTokens * 1000 / (t1 - t0) else 0} tok/s, prefix ${prefixPos} cached) · decode $n tok in ${t2 - t1} ms " +
                String.format("(%.1f tok/s) · total %d ms%s", st.decodeTokPerS, st.totalMs, if (timedOut) " · TIMEOUT" else ""))
            if (timedOut && reply.length < 20) return QcAdvisor.ruleAdvice(findings, trend).copy(source = "rules (Llama timeout)") to st
            val adv = QcAdvisor.parse(reply.toString(), findings, trend)
            if (adv.grounded) cache[key] = adv
            return adv to st
        } catch (e: Throwable) {
            stopped = true; watchdog.interrupt()
            android.util.Log.e("BoardCheck", "llama failed", e)
            prefixPos = -1
            return QcAdvisor.ruleAdvice(findings, trend).copy(source = "rules (Llama error)") to null
        }
    }

    fun close() = llm.close()
}
