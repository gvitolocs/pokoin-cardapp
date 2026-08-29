package com.pokoin.dslocalscan

import android.content.Context
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * CollectorVision Milo: MobileViT-XXS ArcFace 128-d, input 448×448 NCHW ImageNet.
 */
object MiloEmbedder {
    private const val TAG = "MiloEmbedder"
    private const val ASSET = "models/milo.onnx"
    private const val SIZE = 448
    private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)

    @Volatile var available: Boolean = false
        private set

    private var env: OrtEnvironment? = null
    private var session: OrtSession? = null
    private val lock = Any()

    fun init(context: Context) {
        synchronized(lock) {
            if (session != null) return
            try {
                val file = YoloDetector.materialize(context, ASSET)
                val opts = OrtSession.SessionOptions().apply {
                    setIntraOpNumThreads(4)
                    setInterOpNumThreads(1)
                }
                env = OrtEnvironment.getEnvironment()
                session = env!!.createSession(file.absolutePath, opts)
                available = true
                Log.i(TAG, "loaded ${file.length()} bytes")
            } catch (t: Throwable) {
                Log.w(TAG, "Milo ONNX unavailable", t)
                available = false
            }
        }
    }

    fun embed(bgrCrop: Mat): FloatArray? {
        val sess = session ?: return null
        val environment = env ?: return null
        val rgb = Mat()
        val resized = Mat()
        val f32 = Mat()
        try {
            Imgproc.cvtColor(bgrCrop, rgb, Imgproc.COLOR_BGR2RGB)
            Imgproc.resize(rgb, resized, Size(SIZE.toDouble(), SIZE.toDouble()))
            resized.convertTo(f32, CvType.CV_32FC3, 1.0 / 255.0)
            val hw = SIZE * SIZE
            val planar = FloatArray(3 * hw)
            val packed = FloatArray(hw * 3)
            f32.get(0, 0, packed)
            for (i in 0 until hw) {
                for (c in 0 until 3) {
                    planar[c * hw + i] = (packed[i * 3 + c] - MEAN[c]) / STD[c]
                }
            }
            val inputName = sess.inputNames.first()
            val direct = java.nio.ByteBuffer.allocateDirect(planar.size * 4).order(java.nio.ByteOrder.nativeOrder())
            val fb = direct.asFloatBuffer()
            fb.put(planar)
            fb.rewind()
            val tensor = OnnxTensor.createTensor(environment, fb, longArrayOf(1, 3, SIZE.toLong(), SIZE.toLong()))
            synchronized(lock) {
                tensor.use {
                    sess.run(mapOf(inputName to it)).use { result ->
                        val tensorOut = result[0] as OnnxTensor
                        val fbOut = tensorOut.floatBuffer
                        val raw = FloatArray(128)
                        fbOut.get(raw)
                        return l2(raw)
                    }
                }
            }
        } catch (t: Throwable) {
            Log.w(TAG, "embed failed", t)
            return null
        } finally {
            rgb.release()
            resized.release()
            f32.release()
        }
    }

    private fun l2(v: FloatArray): FloatArray {
        var s = 0f
        for (x in v) s += x * x
        val n = kotlin.math.sqrt(s).coerceAtLeast(1e-8f)
        return FloatArray(v.size) { i -> v[i] / n }
    }
}
