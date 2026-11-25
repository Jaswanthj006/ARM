package com.boardcheck.app

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor

/**
 * "Is there a circuit board in this photo?" (ml/train_gate.py, pcb_gate.pte).
 *
 * Runs before the defect detector. Without it, a photo of a desk or keyboard
 * has no defects and would be reported as a passing board. Uses the detector's
 * own backbone + a logistic-regression head, 320 px letterboxed on black.
 */
class PcbGate(modelPath: String) {
    companion object {
        const val SIZE = 320; const val THRESHOLD = 0.5f; const val BLOCK = 0.05f
        // A PASS (no defects) needs a clear board in view. Real clean boards score 0.97-0.99 on the phone and the
        // green-board video frames >= 0.77; a room photo scored 0.48, so 0.70 leaves a margin on both sides.
        const val PASS_MIN = 0.70f
    }

    private val module = Module.load(modelPath, Module.LOAD_MODE_MMAP, Detector.THREADS)
    private val input = FloatArray(3 * SIZE * SIZE)
    private val pixels = IntArray(SIZE * SIZE)

    /** prob = P(photo shows a PCB). board = where the board is, in photo pixels (null = use the whole photo). */
    data class Result(val prob: Float, val board: android.graphics.Rect?)

    /** Probability that the photo shows a PCB. */
    fun score(photo: Bitmap): Float = check(photo).prob

    /**
     * Downscale like training (OpenCV INTER_AREA averages pixels). One bilinear jump from 1920 px to 320 px
     * skips pixels and aliases fine patterns: on the phone a room photo scored 0.51 instead of 0.33 and
     * passed as a board. Halving steps (each a 2x bilinear average) give nearly the same image as training.
     */
    private fun shrink(src: Bitmap, w: Int, h: Int): Bitmap {
        var b = src
        while (b.width / 2 >= w && b.height / 2 >= h) b = Bitmap.createScaledBitmap(b, b.width / 2, b.height / 2, true)
        return if (b.width == w && b.height == h) b else Bitmap.createScaledBitmap(b, w, h, true)
    }

    fun check(photo: Bitmap): Result {
        val s = SIZE.toFloat() / maxOf(photo.width, photo.height)
        val w = maxOf(1, (photo.width * s).toInt()); val h = maxOf(1, (photo.height * s).toInt())
        val boxed = Bitmap.createBitmap(SIZE, SIZE, Bitmap.Config.ARGB_8888)
        Canvas(boxed).apply { drawColor(Color.BLACK); drawBitmap(shrink(photo, w, h), 0f, 0f, null) }
        boxed.getPixels(pixels, 0, SIZE, 0, 0, SIZE, SIZE)
        val plane = SIZE * SIZE
        for (i in 0 until plane) {
            val p = pixels[i]
            input[i] = ((p shr 16) and 0xFF) / 255f
            input[plane + i] = ((p shr 8) and 0xFF) / 255f
            input[2 * plane + i] = (p and 0xFF) / 255f
        }
        val out = module.forward(EValue.from(Tensor.fromBlob(input, longArrayOf(1, 3, SIZE.toLong(), SIZE.toLong()))))
        val prob = out[0].toTensor().dataAsFloatArray[0]
        if (out.size < 2) return Result(prob, null)                       // older gate without the board map
        // 10x10 map at 320 px (32 px cells). Box around the "board" cells, padded by one cell.
        val map = out[1].toTensor().dataAsFloatArray
        var r0 = 10; var c0 = 10; var r1 = -1; var c1 = -1
        for (i in 0 until 100) if (map[i] >= 0.5f) { val r = i / 10; val c = i % 10
            r0 = minOf(r0, r); c0 = minOf(c0, c); r1 = maxOf(r1, r); c1 = maxOf(c1, c) }
        if (r1 < 0) return Result(prob, null)
        val cell = 32f / s
        val box = android.graphics.Rect(
            maxOf(0, ((c0 - 1) * cell).toInt()), maxOf(0, ((r0 - 1) * cell).toInt()),
            minOf(photo.width, ((c1 + 2) * cell).toInt()), minOf(photo.height, ((r1 + 2) * cell).toInt()))
        // A box covering most of the photo saves nothing: tile the whole photo.
        val frac = box.width().toFloat() * box.height() / (photo.width.toFloat() * photo.height)
        return Result(prob, if (frac > 0.8f || box.width() < 64 || box.height() < 64) null else box)
    }

    fun close() = module.destroy()
}
