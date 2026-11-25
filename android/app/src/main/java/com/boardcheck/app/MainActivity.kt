package com.boardcheck.app

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.os.BatteryManager
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.util.Size
import android.view.Gravity
import android.view.View
import android.view.WindowManager
import android.widget.Button
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.barcode.BarcodeScannerOptions
import com.google.mlkit.vision.barcode.BarcodeScanning
import com.google.mlkit.vision.barcode.common.Barcode
import com.google.mlkit.vision.common.InputImage
import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.File
import java.util.concurrent.Executors

/**
 * BoardCheck operator screen.
 *
 *   camera -> tiles -> YOLO (ExecuTorch) -> defect record -> Llama (ExecuTorch) -> advice
 *
 * Everything runs on this phone's Arm CPU. Results, advice and a low-rate camera
 * stream are sent to the laptop bridge, which feeds the supervisor website.
 */
class MainActivity : AppCompatActivity() {

    companion object {
        const val PHOTO_MAX = 1920         // longest side of the inspected photo -> 12 tiles
        const val BOARD_CROP = false       // tile only the board area from the gate's board map (see analyse)
        const val QR_MAX = 960             // QR sticker scan runs on a smaller copy
        const val IDLE_MS = 60_000L        // stop camera analysis + stream after 60 s with no activity
        const val STREAM_WIDTH = 640       // live frames to the website
        const val STREAM_EVERY_MS = 250L
        const val QR_EVERY_MS = 300L
        const val USB_URL = "ws://127.0.0.1:8765/phone"
        val INK = Color.parseColor("#1A1A1A"); val CREAM = Color.parseColor("#FFFDF0")
        val LAVENDER = Color.parseColor("#EBD9FF"); val MUTED = Color.parseColor("#5F5E5A"); val CREAM_2 = Color.parseColor("#F1EEDD")
        val SEV = mapOf("high" to Color.parseColor("#E8603C"), "medium" to Color.parseColor("#EF9F27"), "low" to Color.parseColor("#5DCAA5"))
        val VERDICT = mapOf("pass" to Color.parseColor("#1D9E75"), "rework" to Color.parseColor("#BA7517"), "reject" to Color.parseColor("#E8603C"))
    }

    private val ui = Handler(Looper.getMainLooper())
    private val ai = Executors.newSingleThreadExecutor()          // YOLO + Llama, one at a time
    private val cam = Executors.newSingleThreadExecutor()
    private var detector: Detector? = null
    private var advisor: Advisor? = null
    private var gate: PcbGate? = null
    private var imageCapture: ImageCapture? = null
    private var analysis: ImageAnalysis? = null
    @Volatile private var lastActive = SystemClock.elapsedRealtime()
    @Volatile private var analysing = false
    private var photoMax = PHOTO_MAX
    private var forceLlama = false
    private var pinBig = false           // keep inference on the big A715 cores (adb --ez pin_big true)
    private var noCache = false          // profiling: run Llama on every board (adb --ez no_cache true)
    /** Defect types on the most recent boards (newest last), for trend-aware root cause. */
    private val history = ArrayDeque<Set<String>>()
    private val thresholds = HashMap(Detector.DEFAULT_THRESHOLDS)
    private val prefs by lazy { getSharedPreferences("boardcheck", Context.MODE_PRIVATE) }
    private lateinit var bridge: BridgeClient
    private val qr by lazy { BarcodeScanning.getClient(BarcodeScannerOptions.Builder().setBarcodeFormats(Barcode.FORMAT_QR_CODE).build()) }

    @Volatile private var busy = false
    @Volatile private var live = true
    @Volatile private var pairing = false
    @Volatile private var lastQr = 0L
    @Volatile private var lastPcb = 1f
    private val qrBusy = java.util.concurrent.atomic.AtomicBoolean(false)
    private var lastSent = 0L
    private var frames = 0; private var fps = 0; private var fpsT = SystemClock.elapsedRealtime()
    private var lastMsPerTile = 0L; private var lastTokS = 0.0

    private lateinit var preview: PreviewView
    private lateinit var overlay: LiveOverlay
    // Live detection on camera frames (one centre tile), shown on the preview and the website.
    private val liveBusy = java.util.concurrent.atomic.AtomicBoolean(false)
    private var detFrames = 0; private var detFps = 0; private var detT = SystemClock.elapsedRealtime(); private var liveMs = 0L
    private lateinit var still: ImageView
    private lateinit var linkPill: TextView
    private lateinit var boardTv: TextView
    private lateinit var verdictTv: TextView
    private lateinit var findingsTv: TextView
    private lateinit var adviceTv: TextView
    private lateinit var statusTv: TextView
    private lateinit var inspectBtn: Button

    private val modelDir by lazy { File(filesDir, "models") }
    /** Golden-board reference (one known-good board of the design), files/golden.jpg. */
    private val golden by lazy { Golden(File(filesDir, "golden.jpg")) }
    @Volatile private var lastPhoto: Bitmap? = null
    /** Findings on the last inspected board (before golden filtering), and when a "save anyway" confirm expires. */
    @Volatile private var lastFindings = 0
    @Volatile private var goldenConfirmUntil = 0L

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        buildUi()
        bridge = BridgeClient(::onLink, ::onCommand)
        bridge.connect(prefs.getString("bridge_url", null) ?: USB_URL)
        loadModels()
        if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) startCamera()
        else requestPermissions(arrayOf(Manifest.permission.CAMERA), 1)
        ui.post(statusTick)
        // Test / gallery mode from adb: inspect a stored board image with the same pipeline.
        //   adb shell am start -n com.boardcheck.app/.MainActivity --es image files/test/<name>.jpg
        handleTestIntent(intent)
    }

    /** adb test mode: --es image files/test/<name>.jpg [--ei photo_max 1440] [--ez no_llama true] */
    private fun handleTestIntent(i: Intent) {
        photoMax = i.getIntExtra("photo_max", PHOTO_MAX)
        forceLlama = i.getBooleanExtra("force_llama", false)
        noCache = i.getBooleanExtra("no_cache", false)
        if (i.hasExtra("pin_big")) pinBig = i.getBooleanExtra("pin_big", false)
        i.getStringExtra("image")?.let { name -> inspectFile(File(filesDir.parentFile, name)) }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        handleTestIntent(intent)
    }

    private fun inspectFile(f: File) {
        if (!f.exists()) return status("No image at ${f.path}")
        busy = true; live = false; pauseAnalysis()
        ai.execute {
            while (detector == null || advisor == null) Thread.sleep(200)   // wait for models on cold start
            val photo = scale(android.graphics.BitmapFactory.decodeFile(f.path), photoMax)
            ui.post { inspectBtn.isEnabled = false; setVerdict("Checking…", CREAM_2); adviceTv.text = ""; findingsTv.text = "" }
            analyse(photo, f.nameWithoutExtension)
        }
    }

    override fun onRequestPermissionsResult(code: Int, perms: Array<out String>, res: IntArray) {
        super.onRequestPermissionsResult(code, perms, res)
        if (res.firstOrNull() == PackageManager.PERMISSION_GRANTED) startCamera() else status("Camera permission is needed to inspect boards.")
    }

    // ---------------------------------------------------------------- UI
    private fun dp(v: Int) = (v * resources.displayMetrics.density).toInt()
    private fun pill(bg: Int, stroke: Int = INK) = GradientDrawable().apply { setColor(bg); cornerRadius = dp(14).toFloat(); setStroke(dp(1), stroke) }

    private lateinit var band: LinearLayout
    private lateinit var reasonTv: TextView

    /** Verdict band: big word on a coloured strip, readable from 1-2 m. */
    private fun setVerdict(text: String, color: Int, reason: String = "") {
        verdictTv.text = text
        band.background = GradientDrawable().apply { setColor(color); cornerRadius = dp(16).toFloat() }
        val light = color == CREAM_2
        verdictTv.setTextColor(if (light) INK else Color.WHITE); boardTv.setTextColor(if (light) MUTED else Color.argb(220, 255, 255, 255))
        reasonTv.setTextColor(if (light) MUTED else Color.WHITE); reasonTv.text = reason
        reasonTv.visibility = if (reason.isEmpty()) View.GONE else View.VISIBLE
    }

    private fun buildUi() {
        val root = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setBackgroundColor(CREAM); setPadding(dp(12), dp(34), dp(12), dp(26)) }

        // Header: name + laptop link pill
        val top = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL; setPadding(dp(4), 0, 0, dp(8)) }
        top.addView(TextView(this).apply { text = "BoardCheck"; textSize = 22f; typeface = Typeface.create(Typeface.SERIF, Typeface.BOLD); setTextColor(INK) }, LinearLayout.LayoutParams(0, -2, 1f))
        linkPill = TextView(this).apply { textSize = 12f; setPadding(dp(10), dp(5), dp(10), dp(5)); setTextColor(INK) }
        top.addView(linkPill)
        root.addView(top)

        // Camera / result image
        val stage = FrameLayout(this).apply { background = pill(Color.BLACK, Color.BLACK); clipToOutline = true }
        preview = PreviewView(this).apply { scaleType = PreviewView.ScaleType.FIT_CENTER }
        still = ImageView(this).apply { visibility = View.GONE; scaleType = ImageView.ScaleType.FIT_CENTER; setBackgroundColor(Color.BLACK) }
        overlay = LiveOverlay(this)
        stage.addView(preview, FrameLayout.LayoutParams(-1, -1)); stage.addView(overlay, FrameLayout.LayoutParams(-1, -1)); stage.addView(still, FrameLayout.LayoutParams(-1, -1))
        root.addView(stage, LinearLayout.LayoutParams(-1, 0, 1.25f))

        // Verdict band
        band = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(dp(16), dp(10), dp(16), dp(10)) }
        val bandRow = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL }
        verdictTv = TextView(this).apply { textSize = 30f; typeface = Typeface.DEFAULT_BOLD; text = "Ready" }
        boardTv = TextView(this).apply { textSize = 13f; text = "Board —"; gravity = Gravity.END }
        bandRow.addView(verdictTv, LinearLayout.LayoutParams(0, -2, 1f)); bandRow.addView(boardTv)
        reasonTv = TextView(this).apply { textSize = 14f }
        band.addView(bandRow); band.addView(reasonTv)
        root.addView(band, LinearLayout.LayoutParams(-1, -2).apply { topMargin = dp(10) })

        // Results sheet: defects + what to do
        val info = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(dp(14), dp(10), dp(14), dp(12)) }
        fun head(t: String) = TextView(this).apply { text = t; textSize = 11f; letterSpacing = 0.08f; typeface = Typeface.DEFAULT_BOLD; setTextColor(MUTED); setPadding(0, dp(6), 0, dp(4)) }
        info.addView(head("DEFECTS"))
        findingsTv = TextView(this).apply { textSize = 15f; setTextColor(INK); setLineSpacing(dp(3).toFloat(), 1f); text = "Aim at the board and tap the shutter." }
        info.addView(findingsTv)
        info.addView(head("WHAT TO DO"))
        adviceTv = TextView(this).apply { textSize = 15f; setTextColor(INK); setLineSpacing(dp(3).toFloat(), 1f); text = "Llama's advice appears here after an inspection." }
        info.addView(adviceTv)
        val sheet = ScrollView(this).apply { addView(info); background = pill(Color.WHITE, Color.parseColor("#1A000000")) }
        root.addView(sheet, LinearLayout.LayoutParams(-1, 0, 1f).apply { topMargin = dp(10) })

        // Bottom: Live | big round shutter | Connect
        val bar = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL; setPadding(0, dp(10), 0, 0) }
        fun side(label: String, onClick: (View) -> Unit) = Button(this).apply {
            text = label; isAllCaps = false; textSize = 14f; setTextColor(INK); background = pill(CREAM_2, Color.TRANSPARENT); setOnClickListener(onClick)
        }
        bar.addView(side("◉ Live") { showLive() }, LinearLayout.LayoutParams(0, dp(48), 1f))
        inspectBtn = Button(this).apply {
            text = "INSPECT"; isAllCaps = false; textSize = 15f; typeface = Typeface.DEFAULT_BOLD; setTextColor(Color.WHITE)
            background = GradientDrawable().apply { shape = GradientDrawable.OVAL; setColor(INK); setStroke(dp(5), LAVENDER) }
            setOnClickListener { inspect(null) }
        }
        bar.addView(inspectBtn, LinearLayout.LayoutParams(dp(96), dp(96)).apply { marginStart = dp(14); marginEnd = dp(14) })
        bar.addView(side("⇄ Connect") { v ->
            android.widget.PopupMenu(this, v).apply {
                menu.add("USB link (cable)").setOnMenuItemClickListener { connectTo(USB_URL); true }
                menu.add("Pair with laptop QR").setOnMenuItemClickListener { pairing = true; showLive(); showLive(); status("Point the camera at the QR code on the laptop."); true }
                menu.add("Use last board as golden reference").setOnMenuItemClickListener { setGolden(); true }
                if (golden.isSet) menu.add("Clear golden reference").setOnMenuItemClickListener { clearGolden(); true }
            }.show()
        }, LinearLayout.LayoutParams(0, dp(48), 1f))
        root.addView(bar)
        statusTv = TextView(this).apply { textSize = 11f; setTextColor(MUTED); gravity = Gravity.CENTER; setPadding(0, dp(6), 0, 0); maxLines = 2 }
        root.addView(statusTv)
        setContentView(root)
        setVerdict("Ready", CREAM_2)
        onLink(false)
    }

    private fun status(s: String) { android.util.Log.i("BoardCheck", "status: $s"); ui.post { statusTv.text = s } }

    /**
     * Saves the last inspected photo as the golden reference (a known-good board of this design).
     * The reference must be defect-free: a real defect on it at the same place as on a tested board would
     * hide that defect (seen on the phone test with a defective reference). So if the last board had
     * findings, the first request only warns; a second request within 10 s (or force=true) saves it.
     */
    private fun setGolden(force: Boolean = false) {
        val p = lastPhoto ?: return status("Inspect a known-good board first, then save it as the golden reference.")
        val now = SystemClock.elapsedRealtime()
        if (lastFindings > 0 && !force && now > goldenConfirmUntil) {
            goldenConfirmUntil = now + 10_000
            val msg = "This board had $lastFindings finding${if (lastFindings == 1) "" else "s"}. The golden reference must be a known-good board: " +
                "a real defect on it would hide the same defect on other boards. Choose it again within 10 s to save anyway."
            status(msg)
            bridge.send(JSONObject().put("type", "golden").put("set", golden.isSet).put("warning", msg))
            return
        }
        goldenConfirmUntil = 0L
        ai.execute {
            golden.save(p)
            status("Golden reference saved. Design features that match it (e.g. trace junctions) will be hidden.")
            bridge.send(JSONObject().put("type", "golden").put("set", true))
        }
    }

    private fun clearGolden() {
        ai.execute { golden.clear(); status("Golden reference cleared."); bridge.send(JSONObject().put("type", "golden").put("set", false)) }
    }

    private fun onLink(up: Boolean) {
        linkPill.text = if (up) "● Laptop connected" else "○ Laptop offline"
        linkPill.background = pill(if (up) LAVENDER else CREAM)
    }

    private fun connectTo(url: String) {
        prefs.edit().putString("bridge_url", url).apply()
        bridge.connect(url)
        status("Connecting to $url …")
    }

    private fun showLive() { live = true; touch(); ui.post { still.visibility = View.GONE } }

    // Speed + heat: camera frame analysis is off while an inspection runs and after IDLE_MS
    // without activity, so YOLO and Llama get the CPU and the phone stays cool.
    private fun touch() { lastActive = SystemClock.elapsedRealtime(); if (!busy) resumeAnalysis() }
    private fun pauseAnalysis() { ui.post { analysis?.clearAnalyzer(); analysing = false } }
    private fun resumeAnalysis() { ui.post { if (!analysing) { analysis?.setAnalyzer(cam, ::onFrame); analysing = true } } }

    /** Android thermal status (0 none … 4 severe … 6 shutdown). */
    private fun thermal(): Int = if (Build.VERSION.SDK_INT >= 29)
        (getSystemService(POWER_SERVICE) as android.os.PowerManager).currentThermalStatus else 0

    // ---------------------------------------------------------------- models
    private fun loadModels() = ai.execute {
        try {
            val t0 = SystemClock.elapsedRealtime()
            status("Loading Llama 3.2 1B and YOLO …")
            // One shared ExecuTorch thread pool for all three models, 2 threads (the big cores).
            android.system.Os.setenv("EXECUTORCH_MAX_THREADS", Advisor.THREADS.toString(), true)
            // v3 (7 classes, trained with real boards) first; older 6-class models still load.
            val yolo = listOf("v3_fp16", "fp16", "fp32", "int8").map { File(modelDir, "pcb_yolo11n_$it.pte") }.first { it.exists() }
            detector = Detector(yolo.path).also { it.warmUp() }
            // v4 board check knows yellow bare-copper, green and red real boards; older gate as fallback.
            listOf("pcb_gate_v5.pte", "pcb_gate_v4.pte", "pcb_gate.pte").map { File(modelDir, it) }.firstOrNull { it.exists() }?.let { gate = PcbGate(it.path); android.util.Log.i("BoardCheck", "board check model: ${it.name}") }
            advisor = Advisor(File(modelDir, "llama1b_8da4w.pte").path, File(modelDir, "tokenizer.json").path)
            status("Models ready in ${SystemClock.elapsedRealtime() - t0} ms · ${if (gate != null) "PCB check · " else ""}${yolo.name} · Llama prefix cache · ${Advisor.THREADS} threads · all on-device")
        } catch (e: Throwable) {
            status("Model load failed: ${e.message}")
        }
    }

    // ---------------------------------------------------------------- camera
    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()
            val p = Preview.Builder().build().also { it.surfaceProvider = preview.surfaceProvider }
            imageCapture = ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
                .setResolutionSelector(ResolutionSelector.Builder().setResolutionStrategy(
                    ResolutionStrategy(Size(1920, 1440), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER)).build())
                .build()
            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .setResolutionSelector(ResolutionSelector.Builder().setResolutionStrategy(
                    ResolutionStrategy(Size(1280, 960), ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER)).build())
                .build().also { analysis = it; it.setAnalyzer(cam, ::onFrame); analysing = true }
            provider.unbindAll()
            provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, p, imageCapture, analysis)
        }, mainExecutor)
    }

    /** Live frames: QR pairing, and a low-rate JPEG stream to the website. */
    private fun onFrame(image: ImageProxy) {
        try {
            frames++
            val now = SystemClock.elapsedRealtime()
            if (now - fpsT >= 1000) { fps = frames; frames = 0; fpsT = now }
            val wantStream = live && !busy && bridge.connected && now - lastSent >= STREAM_EVERY_MS
            // Pairing is automatic: while the laptop is offline, every few frames are checked for the
            // website's QR code, so the operator just points the phone at the screen.
            val wantQr = (pairing || !bridge.connected) && now - lastQr >= QR_EVERY_MS && !qrBusy.get()
            val wantLive = live && !busy && !pairing && detector != null && !liveBusy.get()
            if (!wantQr && !wantStream && !wantLive) return
            val raw = image.toBitmap(); val rot = image.imageInfo.rotationDegrees
            if (wantLive && liveBusy.compareAndSet(false, true)) ai.execute { liveDetect(raw, rot) }
            if (wantQr && qrBusy.compareAndSet(false, true)) {
                lastQr = now
                qr.process(InputImage.fromBitmap(scale(raw, QR_MAX), rot))
                    .addOnSuccessListener { codes ->
                        val code = codes.firstOrNull { it.rawValue?.startsWith("ws://") == true }?.rawValue
                        if (code != null && (code != bridge.target || pairing)) { pairing = false; connectTo(code) }
                    }
                    .addOnCompleteListener { qrBusy.set(false) }
            }
            if (wantStream) { lastSent = now; bridge.sendFrame(jpeg(scale(rotate(raw, rot), STREAM_WIDTH), 60)) }
        } finally { image.close() }
    }

    /** Runs on the AI thread (same thread as inspections, so YOLO is never used twice at once). */
    private fun liveDetect(frame: Bitmap, rot: Int) {
        try {
            if (busy || !live) { overlay.clear(); return }
            val r = detector!!.detectLive(frame, rot, thresholds)
            val fw = r.w.toFloat(); val fh = r.h.toFloat()
            liveMs = r.ms; detFrames++
            val now = SystemClock.elapsedRealtime()
            if (now - detT >= 1000) { detFps = detFrames; detFrames = 0; detT = now }
            overlay.show(r, r.w, r.h, detFps)
            if (bridge.connected) bridge.send(JSONObject().put("type", "live").put("fps", detFps).put("ms", r.ms)
                .put("zone", JSONObject().put("x", r.zone.left / fw).put("y", r.zone.top / fh)
                    .put("w", r.zone.width() / fw).put("h", r.zone.height() / fh))
                .put("items", JSONArray().apply { r.boxes.forEach { b -> put(JSONObject().put("type", b.type).put("conf", b.conf.toDouble())
                    .put("severity", QcAdvisor.severity(b.type))
                    .put("x", b.x1 / fw).put("y", b.y1 / fh).put("w", b.w / fw).put("h", b.h / fh)) } }))
        } catch (e: Throwable) {
            android.util.Log.e("BoardCheck", "live detect failed", e)
        } finally { liveBusy.set(false) }
    }

    /** Draws the live zone, live boxes and the detection FPS over the camera preview (FIT_CENTER mapping). */
    inner class LiveOverlay(ctx: android.content.Context) : View(ctx) {
        @Volatile private var res: Detector.Live? = null
        private var fw = 1; private var fh = 1; private var f = 0
        private val zoneP = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE; strokeWidth = 3f; color = Color.argb(200, 255, 255, 255)
            pathEffect = android.graphics.DashPathEffect(floatArrayOf(18f, 12f), 0f) }
        private val boxP = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE; strokeWidth = 5f }
        private val txt = Paint(Paint.ANTI_ALIAS_FLAG).apply { textSize = 34f; typeface = Typeface.DEFAULT_BOLD; color = Color.WHITE; setShadowLayer(4f, 0f, 0f, Color.BLACK) }
        fun show(r: Detector.Live, w: Int, h: Int, fps: Int) { res = r; fw = w; fh = h; f = fps; postInvalidate() }
        fun clear() { res = null; postInvalidate() }
        override fun onDraw(c: Canvas) {
            val r = res ?: return
            if (!live || still.visibility == View.VISIBLE) return
            val s = minOf(width.toFloat() / fw, height.toFloat() / fh)
            val ox = (width - fw * s) / 2; val oy = (height - fh * s) / 2
            fun m(x: Float, y: Float, x2: Float, y2: Float) = RectF(ox + x * s, oy + y * s, ox + x2 * s, oy + y2 * s)
            c.drawRoundRect(m(r.zone.left, r.zone.top, r.zone.right, r.zone.bottom), 16f, 16f, zoneP)
            r.boxes.forEach { b ->
                boxP.color = SEV[QcAdvisor.SOP[b.type]?.severity] ?: Color.WHITE
                val rr = m(b.x1, b.y1, b.x2, b.y2); c.drawRoundRect(rr, 8f, 8f, boxP)
                c.drawText(b.type.replace('_', ' '), rr.left, rr.top - 8f, txt)
            }
            c.drawText("LIVE · $f FPS · ${r.boxes.size} defect${if (r.boxes.size == 1) "" else "s"} in zone", ox + 20f, oy + 48f, txt)
        }
    }

    private fun rotate(b: Bitmap, deg: Int): Bitmap =
        if (deg == 0) b else Bitmap.createBitmap(b, 0, 0, b.width, b.height, Matrix().apply { postRotate(deg.toFloat()) }, true)

    private fun scale(b: Bitmap, longest: Int): Bitmap {
        val s = longest.toFloat() / maxOf(b.width, b.height)
        return if (s >= 1f) b else Bitmap.createScaledBitmap(b, (b.width * s).toInt(), (b.height * s).toInt(), true)
    }

    private fun jpeg(b: Bitmap, q: Int): ByteArray = ByteArrayOutputStream().also { b.compress(Bitmap.CompressFormat.JPEG, q, it) }.toByteArray()

    // ---------------------------------------------------------------- inspection
    /** Called by the Inspect button or by the website ("capture", optionally mode=verify). */
    private fun inspect(boardHint: String?) {
        val ic = imageCapture ?: return status("Camera not ready.")
        if (busy) return
        if (detector == null || advisor == null) return status("Models are still loading …")
        busy = true; live = false; pauseAnalysis(); lastActive = SystemClock.elapsedRealtime()
        ui.post {
            inspectBtn.isEnabled = false; setVerdict("Checking…", CREAM_2, "Detecting defects on the phone…")
            findingsTv.text = ""; adviceTv.text = ""
        }
        ic.takePicture(cam, object : ImageCapture.OnImageCapturedCallback() {
            override fun onCaptureSuccess(image: ImageProxy) {
                val photo = scale(rotate(image.toBitmap(), image.imageInfo.rotationDegrees), photoMax)
                image.close()
                ai.execute { analyse(photo, boardHint) }
            }
            override fun onError(e: ImageCaptureException) { busy = false; touch(); status("Capture failed: ${e.message}"); ui.post { inspectBtn.isEnabled = true } }
        })
    }

    private fun region(cx: Float, cy: Float, w: Int, h: Int): String {
        val col = when { cx < w / 3f -> "left"; cx < 2 * w / 3f -> "centre"; else -> "right" }
        val row = when { cy < h / 3f -> "top"; cy < 2 * h / 3f -> "middle"; else -> "bottom" }
        return if (row == "middle" && col == "centre") "centre of the board" else "$row-$col of the board"
    }

    private fun analyse(photo: Bitmap, boardHint: String?) {
        try {
            val t0 = SystemClock.elapsedRealtime()
            // Board ID from the serial QR sticker (traceability), else a timestamp ID.
            val qrId = runCatching { Tasks.await(qr.process(InputImage.fromBitmap(scale(photo, QR_MAX), 0))) }.getOrNull()
                ?.firstOrNull { it.rawValue?.startsWith("ws://") == false }?.rawValue
            val boardId = qrId ?: boardHint ?: "BC-" + (System.currentTimeMillis() / 1000 % 1_000_000).toString().padStart(6, '0')
            val idSource = if (qrId != null) "qr" else if (boardHint != null) "manual" else "auto"

            // Step 1: is there a PCB at all? Otherwise "no defects" would wrongly read as Pass.
            // Keep the last few inspected photos on the phone (files/inspect/) for checking real-board results.
            runCatching {
                val dir = File(filesDir, "inspect").apply { mkdirs() }
                File(dir, "$boardId.jpg").writeBytes(jpeg(photo, 92))
                dir.listFiles()?.sortedBy { it.lastModified() }?.dropLast(30)?.forEach { it.delete() }
            }
            Annotate.begin(Annotate.CH_INSPECT, Annotate.RED, "Inspect $boardId")
            if (pinBig) android.util.Log.i("BoardCheck", "pinned to ${Affinity.pinBig(true)} big cores")
            val g = Annotate.span(Annotate.CH_STAGE, Annotate.RED, "Board check (gate)") { gate?.check(photo) }
            val pcbProb = g?.prob ?: 1f
            lastPcb = pcbProb
            // "No board" result: nothing is logged as a pass. Used when the view is clearly empty (before
            // detection) and when the board check is unsure (< THRESHOLD) and no defect was found either:
            // a room or desk with no defects must never read as PASS (seen on the phone: a room scored 0.33).
            fun noBoard(stage: String) {
                Annotate.end(Annotate.CH_INSPECT); if (pinBig) Affinity.pinBig(false)
                val sent = scale(photo, 1280)
                bridge.sendFrame(jpeg(sent, 80))
                bridge.send(JSONObject().put("type", "not_pcb").put("board_id", boardId).put("pcb_prob", pcbProb.toDouble())
                    .put("width", sent.width).put("height", sent.height))
                ui.post {
                    still.setImageBitmap(photo); still.visibility = View.VISIBLE
                    boardTv.text = "No board detected"
                    setVerdict("No board", CREAM_2, "No circuit board in view")
                    findingsTv.text = String.format("Board check: %.0f%% confident this is a circuit board, below the %.0f%% needed.", pcbProb * 100, PcbGate.PASS_MIN * 100)
                    adviceTv.text = "Place the circuit board flat under the camera so it fills most of the frame, then press Inspect."
                }
                status(String.format("PCB check %.2f → not a board · %s · %d ms", pcbProb, stage, SystemClock.elapsedRealtime() - t0))
            }
            if (pcbProb < PcbGate.BLOCK) { noBoard("defect detection skipped"); return }

            val quality = Annotate.span(Annotate.CH_STAGE, Annotate.CYAN, "Photo quality") { quality(photo) }

            // Board crop: tile only the area the gate's board map points at (fewer tiles, same pixels).
            // Off by default: on real phone frames the board usually fills most of the photo, so the
            // saving was small, and on synthetic tests the box cut the board 6 % of the time.
            val crop = if (BOARD_CROP) g?.board else null
            val area = if (crop != null) Bitmap.createBitmap(photo, crop.left, crop.top, crop.width(), crop.height()) else photo
            val raw = Annotate.span(Annotate.CH_STAGE, Annotate.GREEN, "YOLO11n defect detection") { detector!!.detect(area, thresholds) }
            val ox = crop?.left?.toFloat() ?: 0f; val oy = crop?.top?.toFloat() ?: 0f
            val rank = mapOf("critical" to 0, "major" to 1, "minor" to 2)
            // Unsure there is a board and nothing found: say "No board", never PASS.
            if (raw.boxes.isEmpty() && pcbProb < PcbGate.PASS_MIN) { noBoard("no defects, but the board check is unsure"); return }
            lastPhoto = photo; lastFindings = raw.boxes.size
            // Golden board: hide detections that match the known-good reference (design features, not defects).
            val placed = raw.boxes.map { it.copy(x1 = it.x1 + ox, y1 = it.y1 + oy, x2 = it.x2 + ox, y2 = it.y2 + oy) }
            val tGold = SystemClock.elapsedRealtime()
            val gScores = if (golden.isSet) Annotate.span(Annotate.CH_STAGE, Annotate.YELLOW, "Golden compare") { golden.scores(photo, placed) } else FloatArray(placed.size)
            val kept = placed.filterIndexed { i, _ -> gScores[i] < Golden.MATCH }
            val goldenHidden = placed.size - kept.size
            if (golden.isSet) android.util.Log.i("BoardCheck", "golden: ${placed.size} boxes, $goldenHidden hidden, ${SystemClock.elapsedRealtime() - tGold} ms")
            // Worst first: critical, then major, then minor; by confidence inside each (numbers match on photo, list, website).
            val det = raw.copy(boxes = kept.sortedWith(compareBy<Detector.Box> { rank[QcAdvisor.severity(it.type)] }.thenByDescending { it.conf }))
            android.util.Log.i("BoardCheck", "board crop: ${crop ?: "none (whole photo)"} of ${photo.width}x${photo.height} -> ${det.tiles} tiles")
            lastMsPerTile = det.msPerTile
            val tDetect = SystemClock.elapsedRealtime()
            val findings = det.boxes.map { b ->
                QcAdvisor.Finding(b.type, b.conf, region((b.x1 + b.x2) / 2 - ox, (b.y1 + b.y2) / 2 - oy, area.width, area.height))
            }
            val decision = QcAdvisor.decision(findings)
            val annotated = drawBoxes(photo, det.boxes)
            ui.post { showResult(annotated, boardId, decision, det, findings, goldenHidden) }
            if (decision == "reject") vibrate()

            // Send the inspected photo (as the current frame) and the defect record to the website.
            val sent = scale(photo, 1280)
            val k = sent.width.toFloat() / photo.width
            bridge.sendFrame(jpeg(sent, 80))
            bridge.send(JSONObject().put("type", "detections").put("board_id", boardId).put("board_id_source", idSource)
                .put("width", sent.width).put("height", sent.height).put("verdict", decision)
                .put("detect_ms", tDetect - t0).put("tiles", det.tiles).put("ms_per_tile", det.msPerTile)
                .put("quality", quality ?: JSONObject.NULL).put("pcb_prob", pcbProb.toDouble())
                .put("golden", golden.isSet).put("golden_hidden", goldenHidden)
                .put("items", JSONArray().apply {
                    det.boxes.forEachIndexed { i, b -> put(JSONObject().put("type", b.type).put("conf", b.conf.toDouble())
                        .put("severity", QcAdvisor.severity(b.type))
                        .put("x", b.x1 * k).put("y", b.y1 * k).put("w", b.w * k).put("h", b.h * k).put("near", findings[i].location)) }
                }))

            // Llama advice, streamed to the phone screen and the website.
            // Heat guard: when Android reports severe throttling, Llama would crawl (~3 tok/s),
            // so the operator gets the rule-based advice from the repair guide instead.
            // Trend-aware cause: a defect type on 3+ of the last 5 boards points at the process, not this board.
            val trend = trend(findings)
            history.addLast(findings.map { it.type }.toSet()); while (history.size > 10) history.removeFirst()
            advisor!!.useCache = !noCache
            val (advice, st) = advisor!!.advise(boardId, findings, trend) { tok ->
                bridge.send(JSONObject().put("type", "advice_token").put("text", tok))
                ui.post { adviceTv.append(tok) }
            }
            if (findings.isEmpty()) bridge.send(JSONObject().put("type", "advice_token").put("text", advice.summary))
            st?.let { lastTokS = it.decodeTokPerS }
            val total = SystemClock.elapsedRealtime() - t0
            Annotate.end(Annotate.CH_INSPECT); if (pinBig) Affinity.pinBig(false)
            bridge.send(JSONObject().put("type", "advice_done").put("verdict", decision).put("ms", total)
                .put("advice", advice.toJson()).put("source", if (findings.isEmpty()) "rules (clean)" else advice.source)
                .put("detect_ms", tDetect - t0))
            ui.post {
                adviceTv.text = buildString {
                    append(advice.summary)
                    advice.steps.forEachIndexed { i, s -> append("\n${i + 1}. $s") }
                    if (advice.cause.isNotEmpty()) append("\nLikely cause: ${advice.cause}")
                    if (advice.trend != null) append("\nTrend: ${advice.trend}")
                    if (quality != null) append("\n⚠ Photo: $quality")
                }
            }
            status(String.format("%d tiles · YOLO %d ms/tile · %s · total %.1f s · on-device",
                det.tiles, det.msPerTile, st?.let { String.format("Llama %.1f tok/s", it.decodeTokPerS) } ?: if (findings.isEmpty()) "Llama skipped (clean)" else advice.source, total / 1000.0))
        } catch (e: Throwable) {
            status("Inspection failed: ${e.javaClass.simpleName}: ${e.message}")
        } finally {
            busy = false; touch()
            ui.post { inspectBtn.isEnabled = true }
        }
    }

    /** Types on >= 3 of the last 5 boards that are also on this board, e.g. "spur on 4 of the last 5 boards". */
    private fun trend(findings: List<QcAdvisor.Finding>): String? {
        val recent = history.toList().takeLast(4)
        if (recent.size < 2) return null
        return findings.map { it.type }.distinct().map { t -> t to 1 + recent.count { t in it } }
            .filter { it.second >= 3 }.maxByOrNull { it.second }
            ?.let { "${it.first.replace('_', ' ')} on ${it.second} of the last ${recent.size + 1} boards" }
    }

    /** Photo quality check (blur, light). Returns a warning for the operator, or null when the photo is fine. */
    private fun quality(photo: Bitmap): String? {
        val s = scale(photo, 320); val w = s.width; val h = s.height
        val px = IntArray(w * h); s.getPixels(px, 0, w, 0, 0, w, h)
        val g = IntArray(px.size) { val c = px[it]; ((c shr 16 and 255) * 3 + (c shr 8 and 255) * 6 + (c and 255)) / 10 }
        val mean = g.average()
        var sum = 0.0; var sq = 0.0; var n = 0; var bright = 0
        for (y in 1 until h - 1) for (x in 1 until w - 1) {
            val i = y * w + x
            val l = (4 * g[i] - g[i - 1] - g[i + 1] - g[i - w] - g[i + w]).toDouble()
            sum += l; sq += l * l; n++
            if (g[i] > 250) bright++
        }
        val sharp = sq / n - (sum / n) * (sum / n)
        android.util.Log.i("BoardCheck", String.format("photo quality: sharpness %.0f, brightness %.0f, glare %.1f%%", sharp, mean, bright * 100.0 / n))
        return when {
            mean < 45 -> "too dark, add light and re-inspect"
            bright * 100.0 / n > 8 -> "glare on the board, tilt it or move the light"
            sharp < 40 -> "blurry, hold the phone steady and re-inspect"
            else -> null
        }
    }

    private fun drawBoxes(photo: Bitmap, boxes: List<Detector.Box>): Bitmap {
        val out = photo.copy(Bitmap.Config.ARGB_8888, true)
        val c = Canvas(out)
        val stroke = maxOf(3f, photo.width / 320f)
        val text = Paint(Paint.ANTI_ALIAS_FLAG).apply { textSize = stroke * 6; typeface = Typeface.DEFAULT_BOLD; color = INK }
        boxes.forEachIndexed { i, b ->
            val col = SEV[QcAdvisor.SOP[b.type]?.severity] ?: Color.WHITE
            c.drawRoundRect(RectF(b.x1, b.y1, b.x2, b.y2), stroke * 2, stroke * 2,
                Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE; strokeWidth = stroke; color = col })
            val label = "${i + 1} ${b.type.replace('_', ' ')} ${(b.conf * 100).toInt()}%"
            val tw = text.measureText(label) + stroke * 3
            c.drawRoundRect(RectF(b.x1, b.y1 - stroke * 8, b.x1 + tw, b.y1 - stroke), stroke, stroke, Paint().apply { color = col })
            c.drawText(label, b.x1 + stroke * 1.5f, b.y1 - stroke * 2.6f, text)
        }
        return out
    }

    private fun showResult(img: Bitmap, boardId: String, decision: String, det: Detector.Result, findings: List<QcAdvisor.Finding>, goldenHidden: Int = 0) {
        still.setImageBitmap(img); still.visibility = View.VISIBLE
        boardTv.text = "Board $boardId"
        val crit = findings.count { QcAdvisor.severity(it.type) == "critical" }; val maj = findings.count { QcAdvisor.severity(it.type) == "major" }
        val reason = if (findings.isEmpty()) "No defects found" else listOf(crit to "critical", maj to "major", findings.size - crit - maj to "minor")
            .filter { it.first > 0 }.joinToString(" · ") { "${it.first} ${it.second}" } + String.format("  ·  %.1f s", det.totalMs / 1000.0) +
            (if (lastPcb < PcbGate.THRESHOLD) "  ·  board unclear, check by eye" else "")
        setVerdict(decision.uppercase(), VERDICT[decision] ?: INK, reason)
        // One row per defect: coloured severity tag, type, where, confidence.
        val sb = android.text.SpannableStringBuilder()
        if (findings.isEmpty()) sb.append("No defects found. The board can move to the next station.")
        findings.forEachIndexed { i, f ->
            if (i > 0) sb.append("\n")
            val sev = QcAdvisor.severity(f.type); val col = SEV[QcAdvisor.SOP[f.type]?.severity] ?: MUTED
            val st = sb.length; sb.append(" ${sev.uppercase()} ")
            sb.setSpan(android.text.style.BackgroundColorSpan(col), st, sb.length, 0)
            sb.setSpan(android.text.style.ForegroundColorSpan(Color.WHITE), st, sb.length, 0)
            sb.setSpan(android.text.style.RelativeSizeSpan(0.75f), st, sb.length, 0)
            sb.setSpan(android.text.style.StyleSpan(Typeface.BOLD), st, sb.length, 0)
            val t = sb.length; sb.append("  ${i + 1}. ${f.type.replace('_', ' ').replaceFirstChar { it.uppercase() }}")
            sb.setSpan(android.text.style.StyleSpan(Typeface.BOLD), t, sb.length, 0)
            val m = sb.length; sb.append("  ${f.location} · ${(f.conf * 100).toInt()}%")
            sb.setSpan(android.text.style.ForegroundColorSpan(MUTED), m, sb.length, 0)
        }
        if (det.suppressed > 0) { val m = sb.length; sb.append("\n${det.suppressed} low-confidence detections hidden"); sb.setSpan(android.text.style.ForegroundColorSpan(MUTED), m, sb.length, 0) }
        if (goldenHidden > 0) { val m = sb.length; sb.append("\n$goldenHidden matched the golden board (design features) and were hidden"); sb.setSpan(android.text.style.ForegroundColorSpan(MUTED), m, sb.length, 0) }
        findingsTv.text = sb
    }

    private fun vibrate() {
        val v = if (Build.VERSION.SDK_INT >= 31) (getSystemService(VIBRATOR_MANAGER_SERVICE) as VibratorManager).defaultVibrator
        else @Suppress("DEPRECATION") getSystemService(VIBRATOR_SERVICE) as Vibrator
        v.vibrate(VibrationEffect.createWaveform(longArrayOf(0, 180, 120, 180), -1))
    }

    // ---------------------------------------------------------------- website link
    private fun onCommand(cmd: JSONObject) {
        when (cmd.optString("cmd")) {
            "capture" -> { touch(); inspect(cmd.optString("board_id").ifEmpty { null }) }
            "start" -> showLive()
            "wake" -> touch()
            "stop" -> live = false
            "golden_set" -> setGolden(cmd.optBoolean("force", false))
            "golden_clear" -> clearGolden()
            "thresholds" -> cmd.optJSONObject("thresholds")?.let { t ->
                Detector.DEFAULT_THRESHOLDS.keys.forEach { k -> thresholds[k] = t.optDouble(k, Detector.DEFAULT_THRESHOLDS[k]!!.toDouble()).toFloat() }
                status("Thresholds updated from the website.")
            }
        }
    }

    private val statusTick = object : Runnable {
        override fun run() {
            val bat = registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
            val pct = bat?.let { it.getIntExtra(BatteryManager.EXTRA_LEVEL, 0) * 100 / maxOf(1, it.getIntExtra(BatteryManager.EXTRA_SCALE, 100)) }
            val temp = bat?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, 0)?.div(10)
            bridge.send(JSONObject().put("type", "status").put("name", "${Build.MANUFACTURER} ${Build.MODEL}")
                .put("fps", fps).put("det_fps", detFps).put("live_ms", liveMs).put("latency_ms", lastMsPerTile).put("llm_tps", Math.round(lastTokS * 10) / 10.0)
                .put("battery", pct).put("temp", temp).put("thermal", thermal()).put("idle", !analysing)
                .put("golden", golden.isSet)
                .put("thresholds", JSONObject().apply { thresholds.forEach { (k, v) -> put(k, Math.round(v * 100) / 100.0) } })
                .put("model", JSONObject().put("vision", "YOLO11n-PCB v3 (7 classes) FP16 · ExecuTorch XNNPACK · ${Detector.THREADS} threads · ${Detector.TILE}px tiles")
                    .put("llm", "Llama 3.2 1B Instruct 8da4w · ExecuTorch · prefix cache · ${Advisor.THREADS} threads")))
            if (SystemClock.elapsedRealtime() - detT > 1500) { detFps = 0; detFrames = 0 }
            if (analysing && !busy && SystemClock.elapsedRealtime() - lastActive > IDLE_MS) {
                analysis?.clearAnalyzer(); analysing = false; live = false
                status("Idle: camera analysis paused to keep the phone cool. Tap Live view or Inspect to wake.")
            }
            ui.postDelayed(this, 1000)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        ui.removeCallbacks(statusTick)
        bridge.disconnect()
        ai.execute { gate?.close(); detector?.close(); advisor?.close() }
        ai.shutdown(); cam.shutdown()
    }
}
