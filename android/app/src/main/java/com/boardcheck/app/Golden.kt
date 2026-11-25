package com.boardcheck.app

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import java.io.File
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt

/**
 * Golden-board comparison (same algorithm and settings as ml/golden_eval.py).
 *
 * The operator saves one known-good board of the design as the reference. For every detection
 * on a new board, the reference (resized to the photo) is searched around the same place for the
 * detection's neighbourhood (zero-mean normalised cross-correlation, so light changes do not
 * matter; the board may be shifted by up to SEARCH x the photo size). If the box itself then also
 * matches the reference (core score >= MATCH), the "defect" is part of the board design, e.g. a
 * T-junction read as a short, and is hidden. A real defect is not on the reference, so it stays.
 *
 * Measured (ml/results/golden_eval.json, T = 0.95): PKU test boards 140 -> 61 false alarms
 * (recall 0.950 -> 0.935); DefectDet, reference = another physical board of the same design:
 * 23 -> 20 false alarms (recall 0.736 -> 0.722).
 */
class Golden(private val file: File) {
    companion object {
        const val WORK = 1000          // compare at this longest side
        const val SEARCH = 0.04f
        const val PAD = 0.6f
        const val MATCH = 0.95f
        const val CONTEXT_MIN = 0.5f   // neighbourhood not found in the reference -> no evidence -> keep
    }

    private var ref: Bitmap? = null
    val isSet: Boolean get() = file.exists()

    fun save(photo: Bitmap) {
        file.outputStream().use { photo.compress(Bitmap.CompressFormat.JPEG, 95, it) }
        ref = null
    }

    fun clear() { file.delete(); ref = null }

    private fun reference(): Bitmap? = ref ?: if (file.exists()) BitmapFactory.decodeFile(file.path).also { ref = it } else null

    private class Gray(val w: Int, val h: Int, val px: FloatArray) {
        // integral images of the values and squares, for fast window means
        val s = DoubleArray((w + 1) * (h + 1)); val s2 = DoubleArray((w + 1) * (h + 1))
        init {
            for (y in 0 until h) { var r = 0.0; var r2 = 0.0
                for (x in 0 until w) { val v = px[y * w + x].toDouble(); r += v; r2 += v * v
                    s[(y + 1) * (w + 1) + x + 1] = s[y * (w + 1) + x + 1] + r
                    s2[(y + 1) * (w + 1) + x + 1] = s2[y * (w + 1) + x + 1] + r2 } }
        }
        fun sum(a: DoubleArray, x: Int, y: Int, ww: Int, hh: Int): Double {
            val W = w + 1
            return a[(y + hh) * W + x + ww] - a[y * W + x + ww] - a[(y + hh) * W + x] + a[y * W + x]
        }
    }

    private fun gray(b: Bitmap, w: Int, h: Int): Gray {
        val sc = Bitmap.createScaledBitmap(b, w, h, true)
        val p = IntArray(w * h); sc.getPixels(p, 0, w, 0, 0, w, h)
        return Gray(w, h, FloatArray(w * h) { val c = p[it]; ((c shr 16 and 255) * 0.299f + (c shr 8 and 255) * 0.587f + (c and 255) * 0.114f) })
    }

    /** One score per box: how closely the box matches the reference board (>= MATCH -> design feature). */
    fun scores(photo: Bitmap, boxes: List<Detector.Box>): FloatArray {
        val r = reference() ?: return FloatArray(boxes.size)
        if (boxes.isEmpty()) return FloatArray(0)
        val s = WORK.toFloat() / max(photo.width, photo.height)
        val w = max(1, (photo.width * s).roundToInt()); val h = max(1, (photo.height * s).roundToInt())
        val t = gray(photo, w, h); val g = gray(r, w, h)
        return FloatArray(boxes.size) { core(t, g, boxes[it], s) }
    }

    /** ZNCC of the test window at (tx,ty) against the reference window at (rx,ry), size pw x ph. */
    private fun zncc(t: Gray, tx: Int, ty: Int, g: Gray, rx: Int, ry: Int, pw: Int, ph: Int): Float {
        val n = (pw * ph).toDouble()
        val mt = t.sum(t.s, tx, ty, pw, ph) / n; val mr = g.sum(g.s, rx, ry, pw, ph) / n
        val vt = t.sum(t.s2, tx, ty, pw, ph) - n * mt * mt; val vr = g.sum(g.s2, rx, ry, pw, ph) - n * mr * mr
        if (vt <= 1e-6 || vr <= 1e-6) return 0f
        var cross = 0.0
        for (y in 0 until ph) { val a = (ty + y) * t.w + tx; val b = (ry + y) * g.w + rx
            for (x in 0 until pw) cross += t.px[a + x] * g.px[b + x] }
        return ((cross - n * mt * mr) / sqrt(vt * vr)).toFloat()
    }

    private fun core(t: Gray, g: Gray, box: Detector.Box, s: Float): Float {
        val W = t.w; val H = t.h
        val x1 = box.x1 * s; val y1 = box.y1 * s
        val bw = max(box.w * s, 6f); val bh = max(box.h * s, 6f)
        val px = bw * PAD + 8; val py = bh * PAD + 8
        val cx1 = max(0, (x1 - px).toInt()); val cy1 = max(0, (y1 - py).toInt())
        val cx2 = min(W, (x1 + bw + px).toInt()); val cy2 = min(H, (y1 + bh + py).toInt())
        val pw = cx2 - cx1; val ph = cy2 - cy1
        if (pw < 8 || ph < 8) return 0f
        val sx = (SEARCH * W).toInt(); val sy = (SEARCH * H).toInt()
        val rx1 = max(0, cx1 - sx); val ry1 = max(0, cy1 - sy)
        val rx2 = min(W - pw, cx1 + sx); val ry2 = min(H - ph, cy1 + sy)
        if (rx2 < rx1 || ry2 < ry1) return 0f
        // Find where the neighbourhood is in the reference: coarse (step 2), then refine.
        var best = -1f; var bx = rx1; var by = ry1
        var y = ry1
        while (y <= ry2) { var x = rx1
            while (x <= rx2) { val v = zncc(t, cx1, cy1, g, x, y, pw, ph); if (v > best) { best = v; bx = x; by = y }; x += 2 }
            y += 2 }
        for (yy in max(ry1, by - 1)..min(ry2, by + 1)) for (xx in max(rx1, bx - 1)..min(rx2, bx + 1)) {
            val v = zncc(t, cx1, cy1, g, xx, yy, pw, ph); if (v > best) { best = v; bx = xx; by = yy } }
        if (best <= CONTEXT_MIN) return 0f
        // The box itself at the matched place.
        val kx = x1.toInt(); val ky = y1.toInt(); val kw = min(bw.toInt(), W - kx); val kh = min(bh.toInt(), H - ky)
        val ox = bx + (kx - cx1); val oy = by + (ky - cy1)
        if (kw * kh < 16 || ox < 0 || oy < 0 || ox + kw > W || oy + kh > H) return 0f
        val n = (kw * kh).toDouble()
        val mt = t.sum(t.s, kx, ky, kw, kh) / n; val mr = g.sum(g.s, ox, oy, kw, kh) / n
        val st = sqrt(max(0.0, t.sum(t.s2, kx, ky, kw, kh) / n - mt * mt)); val sr = sqrt(max(0.0, g.sum(g.s2, ox, oy, kw, kh) / n - mr * mr))
        // No texture to compare (as in golden_eval.py): both flat -> same plain area; otherwise no evidence.
        if (st * st * n <= 1e-6 || sr * sr * n <= 1e-6) return if (st < 3 && sr < 3) 1f else 0f
        return zncc(t, kx, ky, g, ox, oy, kw, kh)
    }
}
