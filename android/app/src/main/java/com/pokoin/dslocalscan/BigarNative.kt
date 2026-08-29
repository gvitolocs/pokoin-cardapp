package com.pokoin.dslocalscan

import android.content.Context
import android.util.Log
import com.bigar.manager.BigarUtils
import com.bigar.manager.helper.VulkanHelper
import org.opencv.core.Core
import org.opencv.core.Mat
import kotlin.math.sqrt

/**
 * Pokoin-native card engine: bundled `libopencvNative.so` + optional Vulkan 1-NN.
 *
 * Live path matches Dragon Shield 8.0.8 `ScanFragment.lambda$processFrame$11`:
 *   CameraX 4:3 RGBA_8888 (no Java resize, no output-image rotation)
 *   -> `Utils.bitmapToMat` RGBA
 *   -> `BigarUtils.rectangleDetection` (native rotate 90 CW, resize 288×384, HOG)
 *   -> Vulkan `computeMatrixDistance` at threshold 1.5, else CPU L2
 *
 * Stills are display-upright; [displayToSensorRgba] center-crops to 3:4 and rotates
 * 90° CCW so native's +90 yields an upright 3:4 card the way a portrait-held camera does.
 */
object BigarNative {
    private const val TAG = "BigarNative"

    const val ASSET_DIR = "bigar_index"
    const val ASSET_GALLERY = "bigar.bigar"
    const val ASSET_MANIFEST = "latest.json"

    private const val MODE_AUTO = 0
    private const val DEFAULT_THRESHOLD = 1.5f
    private const val MAX_GALLERY_FLOATS = 120_000_000L

    data class ScanResult(
        val cardId: Int,
        val row: Int,
        val source: String,
        val ms: Long,
        val l2: Float = Float.MAX_VALUE,
        val detectRc: Int = 1,
        val accepted: Boolean = true,
        val hogNrm: Float = 0f,
        val vulkanId: Int = -1,
        val inRows: Int = -1,
        val inCols: Int = -1,
        val inChannels: Int = -1,
    )

    @Volatile var available: Boolean = false; private set
    @Volatile var status: String = "not attempted"; private set

    private var galleryRows = -1
    private var galleryDim = -1
    private var galleryMatrix: FloatArray? = null
    @Volatile private var vulkanReady = false
    private val initLock = Any()

    fun tryLoad(): Boolean {
        if (available) return true
        synchronized(initLock) {
            if (available) return true
            return try {
                System.loadLibrary("opencvNative")
                status = "lib loaded; gallery not initialized"
                Log.i(TAG, "libopencvNative loaded (vulkan lib present=${VulkanHelper.INSTANCE.isLibraryLoaded()})")
                true
            } catch (t: Throwable) {
                status = "load failed: ${t.message}"
                Log.w(TAG, "libopencvNative unavailable", t)
                false
            }
        }
    }

    fun initGalleryIndex(context: Context): Boolean {
        tryLoad()
        synchronized(initLock) {
            try {
                val dir = java.io.File(context.filesDir, ASSET_DIR).apply { mkdirs() }
                val manifestIn = context.assets.open("$ASSET_DIR/$ASSET_MANIFEST")
                val manifestOut = java.io.File(dir, ASSET_MANIFEST)
                manifestIn.use { input -> manifestOut.outputStream().use { output -> input.copyTo(output) } }

                val galleryIn = context.assets.open("$ASSET_DIR/$ASSET_GALLERY")
                val shippedLen = galleryIn.available().toLong()
                val galleryOut = java.io.File(dir, ASSET_GALLERY)
                if (galleryOut.length() != shippedLen) {
                    galleryIn.use { input -> galleryOut.outputStream().use { output -> input.copyTo(output) } }
                } else {
                    galleryIn.close()
                }

                if (isLoaded()) {
                    BigarUtils.nativeExpansionFileManager(galleryOut.absolutePath)
                }
                if (tryLoad()) {
                    status = "gallery ready at ${galleryOut.absolutePath} (manifest: ${manifestOut.readText()})"
                    Log.i(TAG, status)
                }
                return true
            } catch (t: Throwable) {
                status = "index init failed: ${t.message}"
                Log.e(TAG, "Gallery index bootstrap failed", t)
                return false
            }
        }
    }

    private fun isLoaded(): Boolean = try {
        BigarUtils.getCpuCapabilities()
        true
    } catch (t: Throwable) {
        false
    }

    fun refreshAvailability(): Boolean {
        val ok = tryLoad() && isLoaded()
        if (ok) {
            try {
                galleryRows = BigarUtils.getTotal()
                galleryDim = BigarUtils.getHogs()
            } catch (t: Throwable) {
                Log.w(TAG, "getTotal/getHogs failed", t)
                return false
            }
        }
        available = ok && galleryRows > 0 && galleryDim > 0
        if (!available) status = "engine unusable (rows=$galleryRows dim=$galleryDim)"
        val m = galleryRows.toLong() * galleryDim
        if (available && m <= MAX_GALLERY_FLOATS) {
            val arr = FloatArray(m.toInt())
            BigarUtils.getValues(arr)
            galleryMatrix = arr
        } else if (available) {
            Log.w(TAG, "gallery matrix $m floats exceeds cap; NN path deferred")
        }
        vulkanReady = false
        if (available && VulkanHelper.INSTANCE.isLibraryLoaded()) {
            try {
                vulkanReady = VulkanHelper.INSTANCE.initialize()
            } catch (t: Throwable) {
                Log.w(TAG, "Vulkan init failed; CPU 1-NN only", t)
            }
        }
        return available
    }

    /**
     * Convert a display-upright still (RGBA) into the CameraX sensor buffer Dragon Shield
     * feeds native: 4:3 landscape with the card on its side, so native's 90° CW yields 3:4 upright.
     */
    fun displayToSensorRgba(display: Mat): Mat {
        val w = display.cols()
        val h = display.rows()
        val crop = if (h * 3 >= w * 4) {
            val newH = (w * 4) / 3
            val y = ((h - newH) / 2).coerceAtLeast(0)
            display.submat(y, (y + newH).coerceAtMost(h), 0, w)
        } else {
            val newW = (h * 3) / 4
            val x = ((w - newW) / 2).coerceAtLeast(0)
            display.submat(0, h, x, (x + newW).coerceAtMost(w))
        }
        val sensor = Mat()
        Core.rotate(crop, sensor, Core.ROTATE_90_COUNTERCLOCKWISE)
        crop.release()
        return sensor
    }

    /**
     * Detect + match. [frame] must be the sensor-oriented RGBA Mat DS builds via
     * `imageProxy.toBitmap()` + `bitmapToMat`. Native mutates a clone (rotate/resize).
     */
    fun scanDisplayRgba(
        display: Mat,
        threshold: Float = DEFAULT_THRESHOLD,
        requireAccept: Boolean = true,
    ): ScanResult? {
        val sensor = displayToSensorRgba(display)
        return try {
            scan(sensor, threshold, requireAccept)
        } finally {
            sensor.release()
        }
    }

    fun scan(
        frame: Mat,
        threshold: Float = DEFAULT_THRESHOLD,
        requireAccept: Boolean = true,
    ): ScanResult? {
        if (!available) return null
        val started = System.currentTimeMillis()
        return try {
            val k = galleryDim
            if (k <= 0) return null
            val outHog = FloatArray(k)
            val work = Mat()
            frame.copyTo(work)
            val inRows = work.rows()
            val inCols = work.cols()
            val inCh = work.channels()
            val rc = try {
                BigarUtils.rectangleDetection(work.nativeObj, MODE_AUTO, outHog)
            } finally {
                work.release()
            }
            val hogNrm = hogNorm(outHog)
            if (rc != 1) {
                Log.d(TAG, "rectangleDetection rc=$rc (no card) in=${inRows}x${inCols}x$inCh")
                return ScanResult(
                    cardId = -1,
                    row = -1,
                    source = "bigar-native",
                    ms = System.currentTimeMillis() - started,
                    detectRc = rc,
                    accepted = false,
                    hogNrm = hogNrm,
                    inRows = inRows,
                    inCols = inCols,
                    inChannels = inCh,
                )
            }
            val (cpuRow, dist) = nearestRow(galleryMatrix, outHog)
            if (cpuRow < 0) {
                Log.d(TAG, "gallery empty")
                return null
            }
            val cpuId = BigarUtils.getMinReturnID(cpuRow)
            val vulkanId = vulkanMatch(outHog, threshold)
            val useVulkan = vulkanReady && vulkanId > 0
            val cardId = if (useVulkan) vulkanId else cpuId
            if (cardId <= 0) {
                Log.d(TAG, "getMinReturnID($cpuRow)=$cardId invalid vulkan=$vulkanId")
                return null
            }
            val source = when {
                useVulkan -> "vulkan"
                vulkanReady -> "cpu-after-vulkan-miss"
                else -> "cpu"
            }
            val accepted = if (vulkanReady && requireAccept) vulkanId > 0 else dist <= threshold
            val hit = ScanResult(
                cardId = cardId,
                row = cpuRow,
                source = source,
                ms = System.currentTimeMillis() - started,
                l2 = dist,
                detectRc = rc,
                accepted = accepted,
                hogNrm = hogNrm,
                vulkanId = vulkanId,
                inRows = inRows,
                inCols = inCols,
                inChannels = inCh,
            )
            if (requireAccept && !hit.accepted) {
                Log.d(TAG, "no gallery row under threshold=$threshold (best l2=${hit.l2} id=${hit.cardId})")
                return null
            }
            hit
        } catch (t: Throwable) {
            Log.w(TAG, "scan failed", t)
            null
        }
    }

    private fun vulkanMatch(query: FloatArray, threshold: Float): Int {
        if (!vulkanReady) return -1
        val gallery = galleryMatrix ?: return -1
        return try {
            val row = VulkanHelper.INSTANCE.computeMatrixDistance(
                gallery, query, galleryRows, galleryDim, threshold,
            )
            if (row < 0) -1 else BigarUtils.getMinReturnID(row)
        } catch (t: Throwable) {
            Log.w(TAG, "Vulkan NN failed; using CPU", t)
            -1
        }
    }

    private fun hogNorm(hog: FloatArray): Float {
        var s = 0.0
        for (v in hog) s += v.toDouble() * v.toDouble()
        return sqrt(s).toFloat()
    }

    private fun nearestRow(gallery: FloatArray?, query: FloatArray): Pair<Int, Float> {
        val m = galleryRows
        if (gallery == null || m <= 0) return -1 to Float.MAX_VALUE
        val dim = galleryDim
        var best = -1
        var bestD = Float.MAX_VALUE
        var p = 0
        for (i in 0 until m) {
            var d2 = 0f
            for (c in 0 until dim) {
                val diff = gallery[p + c] - query[c]
                d2 += diff * diff
            }
            val d = sqrt(d2)
            if (d < bestD) {
                bestD = d
                best = i
            }
            p += dim
        }
        return best to bestD
    }

    fun galleryInfo(): String? = if (!available) null
    else "bigar-native ready: cards=$galleryRows dim=$galleryDim matrix=${galleryMatrix?.size ?: "deferred"} vulkan=$vulkanReady"
}
