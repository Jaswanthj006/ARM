package com.boardcheck.app

import android.graphics.Bitmap
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor

/**
 * PCB defect detector: YOLO11n (.pte, ExecuTorch + XNNPACK on the Arm CPU).
 *
 * Mirrors ml/eval_boards.py: the photo is cut into 640 px tiles with 160 px
 * overlap so small defects stay visible, each tile runs through the model,
 * boxes are moved back to photo coordinates and merged with class-wise NMS,
 * then per-defect confidence thresholds are applied.
 *
 * Model output: (1, 4 + nc, 8400) = cx, cy, w, h (tile pixels) + nc class scores.
 * v3 (pcb_yolo11n_v3_fp16.pte) has nc = 7 (adds missing_pad); older models have nc = 6.
 * nc is read from the output size, so either model works with this code.
 */
class Detector(modelPath: String) {

    data class Box(val cls: Int, val conf: Float, val x1: Float, val y1: Float, val x2: Float, val y2: Float) {
        val type get() = NAMES[cls]
        val w get() = x2 - x1
        val h get() = y2 - y1
    }

    data class Result(val boxes: List<Box>, val tiles: Int, val msPerTile: Long, val totalMs: Long, val suppressed: Int)

    companion object {
        val NAMES = listOf("missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper", "missing_pad")
        const val TILE = 640
        const val ANCHORS = 8400   // 80x80 + 40x40 + 20x20 for a 640 px input
        const val THREADS = 2   // same as Advisor.THREADS: one shared pool, never resized after loading
        const val STRIDE = 480
        const val MIN_CONF = 0.25f
        const val NMS_IOU = 0.4f
        // v3 thresholds from ml/results/phaseB_final.json: tuned on half of the PKU boards plus half of the
        // real-board test photos (PCB-Defect, DefectDet), checked on the other half.
        val DEFAULT_THRESHOLDS = mapOf(
            "missing_hole" to 0.50f, "mouse_bite" to 0.35f, "open_circuit" to 0.55f,
            "short" to 0.50f, "spur" to 0.25f, "spurious_copper" to 0.60f, "missing_pad" to 0.45f,
        )
    }

    // FP16 model: ~2x faster than FP32 on the Dimensity 7200 (ml/results/phone/yolo_bench.json).
    private val module: Module = Module.load(modelPath, Module.LOAD_MODE_MMAP, THREADS)

    /** One dummy tile at load, so the first real board is not slowed by lazy loading (644 → ~110 ms/tile). */
    fun warmUp() { module.forward(EValue.from(Tensor.fromBlob(input, longArrayOf(1, 3, TILE.toLong(), TILE.toLong())))) }
    private val input = FloatArray(3 * TILE * TILE)
    private val pixels = IntArray(TILE * TILE)

    private fun starts(size: Int): List<Int> {
        if (size <= TILE) return listOf(0)
        val s = (0..size - TILE step STRIDE).toMutableList()
        if (s.last() + TILE < size) s += size - TILE
        return s
    }

    fun detect(photo: Bitmap, thresholds: Map<String, Float>): Result {
        val t0 = System.nanoTime()
        val raw = ArrayList<Box>()
        var tiles = 0
        val ys = starts(photo.height); val xs = starts(photo.width); val n = ys.size * xs.size
        for (y0 in ys) for (x0 in xs) {
            raw += Annotate.span(Annotate.CH_TILE, Annotate.GREEN, "YOLO tile ${tiles + 1}/$n") { runTile(photo, x0, y0) }
            tiles++
        }
        android.util.Log.i("BoardCheck", "yolo profile: tiles=$tiles prep=${tPrep / 1_000_000 / maxOf(1, tiles)} fwd=${tFwd / 1_000_000 / maxOf(1, tiles)} post=${tPost / 1_000_000 / maxOf(1, tiles)} ms/tile")
        tPrep = 0; tFwd = 0; tPost = 0
        val merged = Annotate.span(Annotate.CH_STAGE, Annotate.CYAN, "NMS") { nms(raw) }
        val kept = merged.filter { it.conf >= (thresholds[it.type] ?: DEFAULT_THRESHOLDS[it.type] ?: 0.4f) }
        val total = (System.nanoTime() - t0) / 1_000_000
        return Result(kept.sortedByDescending { it.conf }, tiles, if (tiles > 0) total / tiles else 0, total, merged.size - kept.size)
    }

    /** Live preview: one 640 tile from the centre of the frame scaled to LIVE_SCALE (about the inspection
     *  photo's pixel scale, so defects look the same size as in training). Boxes and zone are in frame pixels. */
    data class Live(val boxes: List<Box>, val zone: android.graphics.RectF, val ms: Long, val w: Int, val h: Int)
    /** Live preview: the centre 640x640 square of the camera frame at native resolution (no rescale).
     *  The square is cropped from the UNROTATED frame and only that tile is rotated (a centred square is the
     *  same region either way), which avoids rotating the whole 1280x960 frame every time.
     *  Boxes and zone are returned in rotated (upright) frame pixels; w/h are the upright frame size. */
    fun detectLive(frame: Bitmap, rotation: Int, thresholds: Map<String, Float>): Live {
        val t0 = System.nanoTime()
        val z = minOf(TILE, frame.width, frame.height)
        val tile0 = Bitmap.createBitmap(frame, (frame.width - z) / 2, (frame.height - z) / 2, z, z)
        val tile = if (rotation == 0) tile0 else Bitmap.createBitmap(tile0, 0, 0, z, z, android.graphics.Matrix().apply { postRotate(rotation.toFloat()) }, false)
        val sideways = rotation == 90 || rotation == 270
        val w = if (sideways) frame.height else frame.width; val h = if (sideways) frame.width else frame.height
        val zx = (w - z) / 2f; val zy = (h - z) / 2f
        val raw = runTile(tile, 0, 0)
        tPrep = 0; tFwd = 0; tPost = 0
        val boxes = nms(raw).filter { it.conf >= (thresholds[it.type] ?: DEFAULT_THRESHOLDS[it.type] ?: 0.4f) }
            .map { it.copy(x1 = zx + it.x1, y1 = zy + it.y1, x2 = zx + it.x2, y2 = zy + it.y2) }
        return Live(boxes, android.graphics.RectF(zx, zy, zx + z, zy + z), (System.nanoTime() - t0) / 1_000_000, w, h)
    }

    private var tPrep = 0L; private var tFwd = 0L; private var tPost = 0L

    private fun runTile(photo: Bitmap, x0: Int, y0: Int): List<Box> {
        val a = System.nanoTime()
        val w = minOf(TILE, photo.width - x0)
        val h = minOf(TILE, photo.height - y0)
        java.util.Arrays.fill(pixels, 0xFF000000.toInt())          // black padding for small photos
        photo.getPixels(pixels, 0, TILE, x0, y0, w, h)
        val plane = TILE * TILE
        for (i in 0 until plane) {                                  // RGB, 0..1, CHW (same as training)
            val p = pixels[i]
            input[i] = ((p shr 16) and 0xFF) / 255f
            input[plane + i] = ((p shr 8) and 0xFF) / 255f
            input[2 * plane + i] = (p and 0xFF) / 255f
        }
        val b = System.nanoTime(); tPrep += b - a
        val out = module.forward(EValue.from(Tensor.fromBlob(input, longArrayOf(1, 3, TILE.toLong(), TILE.toLong()))))[0]
            .toTensor().dataAsFloatArray
        val c0 = System.nanoTime(); tFwd += c0 - b
        val n = ANCHORS
        val nc = minOf(out.size / n - 4, NAMES.size)               // 6 (old model) or 7 (v3)
        val boxes = ArrayList<Box>()
        for (i in 0 until n) {
            var best = 0; var score = out[4 * n + i]
            for (c in 1 until nc) { val s = out[(4 + c) * n + i]; if (s > score) { score = s; best = c } }
            if (score < MIN_CONF) continue
            val cx = out[i]; val cy = out[n + i]; val bw = out[2 * n + i]; val bh = out[3 * n + i]
            boxes += Box(best, score, x0 + cx - bw / 2, y0 + cy - bh / 2, x0 + cx + bw / 2, y0 + cy + bh / 2)
        }
        tPost += System.nanoTime() - c0
        return boxes
    }

    private fun iou(a: Box, b: Box): Float {
        val ix = maxOf(0f, minOf(a.x2, b.x2) - maxOf(a.x1, b.x1))
        val iy = maxOf(0f, minOf(a.y2, b.y2) - maxOf(a.y1, b.y1))
        val inter = ix * iy
        val u = a.w * a.h + b.w * b.h - inter
        return if (u > 0) inter / u else 0f
    }

    private fun nms(boxes: List<Box>): List<Box> {
        val keep = ArrayList<Box>()
        for (b in boxes.sortedByDescending { it.conf }) {
            if (keep.none { it.cls == b.cls && iou(it, b) > NMS_IOU }) keep += b
        }
        return keep
    }

    fun close() = module.destroy()
}
