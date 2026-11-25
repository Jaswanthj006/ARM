package com.boardcheck.app

import android.graphics.Color
import android.graphics.Typeface
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.view.Gravity
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import org.json.JSONArray
import org.json.JSONObject
import org.pytorch.executorch.extension.llm.LlmCallback
import org.pytorch.executorch.extension.llm.LlmModule
import java.io.File
import java.util.concurrent.Executors

/**
 * Step 1 screen: run the Llama 3.2 1B QC advisor on the phone with ExecuTorch
 * and measure real speed on this Arm CPU. Camera + YOLO are added next.
 *
 * Model files are copied with adb (run-as, debug build) into the app's private storage:
 *   /data/data/com.boardcheck.app/files/models/llama1b_8da4w.pte
 *   /data/data/com.boardcheck.app/files/models/tokenizer.json
 */
class BenchActivity : AppCompatActivity() {

    companion object { const val DEFAULT_THREADS = 2 }

    private val worker = Executors.newSingleThreadExecutor()
    private val ui = Handler(Looper.getMainLooper())
    private var llm: LlmModule? = null
    private lateinit var status: TextView
    private lateinit var out: TextView

    private val modelDir by lazy { File(filesDir, "models") }
    private val llamaFile by lazy { File(modelDir, "llama1b_8da4w.pte") }
    private val tokenizerFile by lazy { File(modelDir, "tokenizer.json") }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        modelDir.mkdirs()
        val pad = (16 * resources.displayMetrics.density).toInt()
        val root = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(pad, pad * 2, pad, pad); setBackgroundColor(Color.parseColor("#FFFDF0")) }
        root.addView(TextView(this).apply { text = "BoardCheck"; textSize = 30f; setTypeface(Typeface.SERIF); setTextColor(Color.parseColor("#1A1A1A")) })
        status = TextView(this).apply { textSize = 14f; setTextColor(Color.parseColor("#5F5E5A")); setPadding(0, pad / 2, 0, pad / 2) }
        root.addView(status)
        val row = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.START }
        row.addView(button("Load Llama") { loadLlama() })
        row.addView(button("Run QC advice") { runSamples() })
        root.addView(row)
        out = TextView(this).apply { textSize = 13f; setTextColor(Color.parseColor("#1A1A1A")); typeface = Typeface.MONOSPACE; setTextIsSelectable(true) }
        root.addView(ScrollView(this).apply { addView(out) }, LinearLayout.LayoutParams(-1, 0, 1f))
        setContentView(root)
        status.text = "Models: ${if (llamaFile.exists()) "Llama found (${llamaFile.length() / 1_000_000} MB)" else "Llama missing"} · " +
            "${if (tokenizerFile.exists()) "tokenizer found" else "tokenizer missing"}\nCPU cores: ${Runtime.getRuntime().availableProcessors()}"
        // Keep the screen on so Android does not pause the app mid-benchmark.
        window.addFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        // adb shell am start -n com.boardcheck.app/.BenchActivity --ez autorun true
        val threads = intent.getStringExtra("threads")?.split(",")?.mapNotNull { it.trim().toIntOrNull() }
        // adb ... --es yolo_bench "2,4,6"   (YOLO FP32 vs INT8 latency per tile and per board)
        val yoloThreads = intent.getStringExtra("yolo_bench")?.split(",")?.mapNotNull { it.trim().toIntOrNull() }
        if (!yoloThreads.isNullOrEmpty()) { benchYolo(yoloThreads) }
        else if (!threads.isNullOrEmpty()) { loadLlama(); sweep(threads) }
        else if (intent.getBooleanExtra("autorun", false)) {
            // adb ... --ez autorun true --ei nthreads 4
            loadLlama(intent.getIntExtra("nthreads", DEFAULT_THREADS))
            runSamples()
        }
    }

    private fun button(label: String, onClick: () -> Unit) = Button(this).apply {
        text = label; isAllCaps = false; setOnClickListener { onClick() }
    }

    private fun log(s: String) = ui.post { out.append(s + "\n") }

    /**
     * YOLO latency on this phone: one 640x640 tile = one forward pass. A whole
     * board photo (~3000x1600) is 21 tiles with 160 px overlap, so board time
     * ~ 21 x tile time. Compares the FP32 and INT8 .pte at several thread counts.
     */
    private fun benchYolo(threadList: List<Int>) = worker.execute {
        val res = JSONArray()
        val input = org.pytorch.executorch.Tensor.fromBlob(FloatArray(3 * 640 * 640) { (it % 255) / 255f }, longArrayOf(1, 3, 640, 640))
        val ev = org.pytorch.executorch.EValue.from(input)
        for (variant in listOf("fp16", "fp32", "int8")) {
            val f = File(modelDir, "pcb_yolo11n_$variant.pte")
            if (!f.exists()) { log("missing ${f.name}"); continue }
            for (n in threadList) {
                val t0 = SystemClock.elapsedRealtime()
                val m = org.pytorch.executorch.Module.load(f.absolutePath, org.pytorch.executorch.Module.LOAD_MODE_MMAP, n)
                m.forward(ev)   // first call includes method load
                val loadMs = SystemClock.elapsedRealtime() - t0
                repeat(3) { m.forward(ev) }   // warm-up
                val times = LongArray(20) {
                    val t = System.nanoTime(); m.forward(ev); (System.nanoTime() - t) / 1_000_000
                }.sorted()
                val med = times[times.size / 2]
                val out = m.forward(ev)[0].toTensor().shape().joinToString("x")
                log(String.format("YOLO %s · %d threads · load %d ms · %d ms/tile (min %d, max %d) · ~%.1f s per 21-tile board · output %s",
                    variant, n, loadMs, med, times.first(), times.last(), med * 21 / 1000.0, out))
                res.put(JSONObject().put("variant", variant).put("threads", n).put("load_ms", loadMs)
                    .put("median_ms_per_tile", med).put("min_ms", times.first()).put("max_ms", times.last())
                    .put("board_21_tiles_s", med * 21 / 1000.0).put("size_mb", f.length() / 1e6))
                m.destroy()
            }
        }
        File(filesDir, "yolo_bench.json").writeText(JSONObject().put("device", android.os.Build.MODEL)
            .put("soc", android.os.Build.SOC_MODEL).put("runs", res).toString(2))
        log("Saved yolo_bench.json")
    }

    private var llamaThreads = 0

    private fun loadLlama(threads: Int = DEFAULT_THREADS) = worker.execute {
        if (llm != null) { log("Llama already loaded."); return@execute }
        if (!llamaFile.exists() || !tokenizerFile.exists()) { log("Model files missing in ${modelDir.absolutePath}"); return@execute }
        val t0 = SystemClock.elapsedRealtime()
        try {
            // Read by our patched ExecuTorch JNI (executorch_android/boardcheck-threads.patch)
            // when LlmModule is created. Stock ExecuTorch leaves a single thread on this chip.
            android.system.Os.setenv("EXECUTORCH_NUM_THREADS", threads.toString(), true)
            llamaThreads = threads
            // Memory-map the model: weights are read from storage on demand instead of being
            // copied into app memory, so Android can reclaim pages instead of killing the app.
            val cfg = org.pytorch.executorch.extension.llm.LlmModuleConfig.create()
                .modulePath(llamaFile.absolutePath).tokenizerPath(tokenizerFile.absolutePath)
                .temperature(0.0f).loadMode(org.pytorch.executorch.extension.llm.LlmModuleConfig.LOAD_MODE_MMAP).build()
            val m = LlmModule(cfg)
            m.load()
            llm = m
            log("Loaded Llama in ${SystemClock.elapsedRealtime() - t0} ms with $threads CPU threads")
            ui.post { status.append("\nLlama loaded") }
        } catch (e: Throwable) {
            log("Load failed: ${e.javaClass.simpleName}: ${e.message}")
        }
    }

    private var primer: org.pytorch.executorch.Module? = null

    /**
     * LlmModule has no thread setting, but ExecuTorch uses one shared CPU thread
     * pool. Loading a tiny model with Module.load(path, mode, numThreads) resets
     * that pool, so Llama runs with the same number of threads.
     */
    private fun setThreads(n: Int) {
        val f = File(modelDir, "threadpool_primer.pte")
        if (!f.exists()) { log("threadpool_primer.pte missing, keeping default threads"); return }
        primer?.destroy()
        primer = org.pytorch.executorch.Module.load(f.absolutePath, org.pytorch.executorch.Module.LOAD_MODE_FILE, n)
        primer?.forward(org.pytorch.executorch.EValue.from(org.pytorch.executorch.Tensor.fromBlob(FloatArray(8), longArrayOf(1, 8))))
        log("CPU threads set to $n")
    }

    /** Thread sweep from adb: --es threads "1,2,4,6" (2 boards per setting), then the full run on the best. */
    private fun sweep(list: List<Int>) = worker.execute {
        val summary = JSONArray()
        for (n in list) {
            setThreads(n)
            val r = runBoards(QcAdvisor.SAMPLES.take(2), "threads_$n")
            summary.put(JSONObject().put("threads", n).put("runs", r))
        }
        File(filesDir, "thread_sweep.json").writeText(
            JSONObject().put("device", android.os.Build.MODEL).put("soc", android.os.Build.SOC_MODEL)
                .put("sweep", summary).toString(2))
        log("\nSaved thread_sweep.json")
    }

    private fun runSamples() = worker.execute {
        val results = runBoards(QcAdvisor.SAMPLES, "default") ?: return@execute
        File(filesDir, "llama_bench.json").writeText(
            JSONObject().put("device", android.os.Build.MODEL).put("soc", android.os.Build.SOC_MODEL)
                .put("runs", results).toString(2))
        log("\nSaved results to llama_bench.json")
    }

    private fun runBoards(boards: List<Pair<String, List<QcAdvisor.Finding>>>, tag: String): JSONArray? {
        val m = llm ?: run { log("Load Llama first."); return null }
        val results = JSONArray()
        for ((board, findings) in boards) {
            val prompt = QcAdvisor.prompt(board, findings)
            val reply = StringBuilder()
            var stats: String? = null
            m.resetContext()   // each board starts from an empty KV cache
            val t0 = SystemClock.elapsedRealtime()
            m.generate(prompt, 768, object : LlmCallback {
                override fun onResult(result: String) {
                    reply.append(result)
                    if (QcAdvisor.complete(reply.toString())) m.stop()   // stop as soon as all labels are written
                }
                override fun onStats(s: String) { stats = s }
                override fun onError(code: Int, message: String) { log("Llama error $code: $message") }
            }, false)
            val wall = SystemClock.elapsedRealtime() - t0
            val text = reply.toString().replace("<|eot_id|>", "")
            val advice = QcAdvisor.parse(text, findings)
            val st = stats?.let { runCatching { JSONObject(it) }.getOrNull() }
            val gen = st?.optInt("generated_tokens") ?: 0
            val promptTok = st?.optInt("prompt_tokens") ?: 0
            val ttft = st?.let { it.optLong("first_token_ms") - it.optLong("inference_start_ms") } ?: 0
            val decodeMs = st?.let { it.optLong("inference_end_ms") - it.optLong("prompt_eval_end_ms") } ?: 0
            val prefillMs = st?.let { it.optLong("prompt_eval_end_ms") - it.optLong("inference_start_ms") } ?: 0
            val tps = if (decodeMs > 0) gen * 1000.0 / decodeMs else 0.0
            val pps = if (prefillMs > 0) promptTok * 1000.0 / prefillMs else 0.0
            log("\n[$board] ${findings.size} finding(s) → ${advice.decision.uppercase()}")
            log("SUMMARY: ${advice.summary}")
            advice.steps.forEach { log("  STEP: $it") }
            log("CAUSE: ${advice.cause}\nLOG: ${advice.log}")
            log(String.format("speed: prompt %d tok @ %.0f tok/s · first token %d ms · %d tok @ %.1f tok/s · total %d ms",
                promptTok, pps, ttft, gen, tps, wall))
            results.put(JSONObject().put("tag", tag).put("threads", llamaThreads).put("board", board).put("advice", advice.toJson()).put("raw", text)
                .put("prompt_tokens", promptTok).put("prefill_tok_s", pps).put("first_token_ms", ttft)
                .put("generated_tokens", gen).put("decode_tok_s", tps).put("total_ms", wall))
        }
        return results
    }

    override fun onDestroy() {
        super.onDestroy()
        worker.execute { llm?.close() }
        worker.shutdown()
    }
}
