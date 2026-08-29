package com.pokoin.dslocalscan

import android.content.Context
import android.util.Log
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * TCG YOLO11n card detector (float16 TFLite, NHWC 640, output [1,5,8400]).
 * Detect only. Identify is Milo / CLIP.
 */
object YoloDetector {
    private const val TAG = "YoloDetector"
    private const val ASSET = "models/card_detector.tflite"
    private const val SIZE = 640
    private const val ANCHORS = 8400
    private const val CONF = 0.25f
    private const val IOU = 0.45f
    private const val MIN_AREA = 0.01f

    data class Box(val x1: Float, val y1: Float, val x2: Float, val y2: Float, val conf: Float) {
        fun toQuad(): FloatArray = floatArrayOf(x1, y1, x2, y1, x2, y2, x1, y2)
    }

    @Volatile var available: Boolean = false
        private set

    private var interpreter: Interpreter? = null
    private val lock = Any()

    fun init(context: Context) {
        synchronized(lock) {
            if (interpreter != null) return
            try {
                val file = materialize(context, ASSET)
                val opts = Interpreter.Options().apply { setNumThreads(4) }
                interpreter = Interpreter(file, opts)
                available = true
                Log.i(TAG, "loaded ${file.length()} bytes")
            } catch (t: Throwable) {
                Log.w(TAG, "YOLO TFLite unavailable", t)
                available = false
            }
        }
    }

    fun detect(bgr: Mat): List<Box> {
        val interp = interpreter ?: return emptyList()
        val h = bgr.rows()
        val w = bgr.cols()
        if (h < 8 || w < 8) return emptyList()
        val resized = Mat()
        val rgb = Mat()
        val f32 = Mat()
        try {
            Imgproc.resize(bgr, resized, Size(SIZE.toDouble(), SIZE.toDouble()))
            Imgproc.cvtColor(resized, rgb, Imgproc.COLOR_BGR2RGB)
            rgb.convertTo(f32, CvType.CV_32FC3, 1.0 / 255.0)
            val pixels = FloatArray(SIZE * SIZE * 3)
            f32.get(0, 0, pixels)
            val input = ByteBuffer.allocateDirect(pixels.size * 4).order(ByteOrder.nativeOrder())
            input.asFloatBuffer().put(pixels)
            input.rewind()
            val out = Array(1) { Array(5) { FloatArray(ANCHORS) } }
            synchronized(lock) {
                interp.run(input, out)
            }
            return decode(out[0], w, h)
        } finally {
            resized.release()
            rgb.release()
            f32.release()
        }
    }

    private fun decode(out: Array<FloatArray>, origW: Int, origH: Int): List<Box> {
        val sx = origW / SIZE.toFloat()
        val sy = origH / SIZE.toFloat()
        val minArea = MIN_AREA * origW * origH
        val raw = ArrayList<Box>(64)
        for (j in 0 until ANCHORS) {
            val conf = out[4][j]
            if (conf < CONF) continue
            val cx = out[0][j]
            val cy = out[1][j]
            val bw = out[2][j]
            val bh = out[3][j]
            val x1 = ((cx - bw / 2f) * sx).coerceIn(0f, origW.toFloat())
            val y1 = ((cy - bh / 2f) * sy).coerceIn(0f, origH.toFloat())
            val x2 = ((cx + bw / 2f) * sx).coerceIn(0f, origW.toFloat())
            val y2 = ((cy + bh / 2f) * sy).coerceIn(0f, origH.toFloat())
            if ((x2 - x1) * (y2 - y1) < minArea) continue
            raw.add(Box(x1, y1, x2, y2, conf))
        }
        raw.sortByDescending { it.conf }
        val kept = ArrayList<Box>(raw.size)
        for (b in raw) {
            if (kept.all { iou(b, it) <= IOU }) kept.add(b)
        }
        return kept
    }

    private fun iou(a: Box, b: Box): Float {
        val ix1 = maxOf(a.x1, b.x1)
        val iy1 = maxOf(a.y1, b.y1)
        val ix2 = minOf(a.x2, b.x2)
        val iy2 = minOf(a.y2, b.y2)
        val inter = maxOf(0f, ix2 - ix1) * maxOf(0f, iy2 - iy1)
        if (inter <= 0f) return 0f
        val aa = (a.x2 - a.x1) * (a.y2 - a.y1)
        val ba = (b.x2 - b.x1) * (b.y2 - b.y1)
        return inter / (aa + ba - inter)
    }

    internal fun materialize(context: Context, asset: String): File {
        val dest = File(context.filesDir, asset)
        dest.parentFile?.mkdirs()
        if (dest.isFile && dest.length() > 0L) return dest
        context.assets.open(asset).use { input ->
            FileOutputStream(dest).use { input.copyTo(it) }
        }
        return dest
    }
}
