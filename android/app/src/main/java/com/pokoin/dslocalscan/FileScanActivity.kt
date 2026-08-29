package com.pokoin.dslocalscan

import android.app.Activity
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageDecoder
import android.os.Build
import android.os.Bundle
import android.util.Log
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Mat
import java.io.File

/**
 * Headless scan: `am start -n .../.FileScanActivity -e scan_path /path.png [-e scan_mode fast|multi]`
 *
 * FAST: YOLO + Milo → TCGplayer (HOG LAYOUT fallback). MULTI: YOLO boxes + Milo.
 */
class FileScanActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val path = intent.getStringExtra(EXTRA_PATH)
        if (path.isNullOrBlank()) {
            Log.e(TAG, "missing extra $EXTRA_PATH")
            finish()
            return
        }
        setContentView(android.R.layout.simple_list_item_1)
        val mode = ScanMode.fromExtra(intent.getStringExtra(EXTRA_MODE))
        Thread {
            try {
                runScan(path, mode)
            } catch (t: Throwable) {
                Log.e(TAG, "scan crashed", t)
                File("$path.json").writeText("""{"ok":false,"error":${org.json.JSONObject.quote(t.toString())}}""" + "\n")
            } finally {
                runOnUiThread { finish() }
            }
        }.start()
    }

    private fun runScan(path: String, mode: ScanMode) {
        if (!OpenCVLoader.initLocal()) {
            File("$path.json").writeText("""{"ok":false,"error":"opencv"}""" + "\n")
            return
        }
        BigarNative.initGalleryIndex(this)
        BigarNative.refreshAvailability()
        FrameScanner.init(this)
        var n = 0
        while (!MiloIndex.available && !BigarNative.available && n < 50) {
            Thread.sleep(200)
            BigarNative.refreshAvailability()
            n++
        }
        val bmp = loadBitmap(path)
        if (bmp == null) {
            File("$path.json").writeText("""{"ok":false,"error":"decode","path":${org.json.JSONObject.quote(path)}}""" + "\n")
            return
        }
        val rgba = Mat()
        Utils.bitmapToMat(bmp, rgba)
        bmp.recycle()
        val result = try {
            FrameScanner.scanStill(rgba, mode)
        } finally {
            rgba.release()
        }
        val payload = FrameScanner.toJson(path, result)
            .put("available", BigarNative.available)
            .put("status", BigarNative.status)
            .put("info", BigarNative.galleryInfo() ?: org.json.JSONObject.NULL)
        result.lastWarp?.release()
        File("$path.json").writeText(payload.toString() + "\n")
        Log.i(TAG, payload.toString())
    }

    private fun loadBitmap(path: String): Bitmap? {
        val raw = if (Build.VERSION.SDK_INT >= 28) {
            val src = ImageDecoder.createSource(File(path))
            ImageDecoder.decodeBitmap(src) { decoder, _, _ ->
                decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
                decoder.isMutableRequired = true
            }
        } else {
            BitmapFactory.decodeFile(path, BitmapFactory.Options().apply {
                inPreferredConfig = Bitmap.Config.ARGB_8888
            })
        } ?: return null
        return if (raw.config == Bitmap.Config.ARGB_8888) raw else raw.copy(Bitmap.Config.ARGB_8888, true)
    }

    companion object {
        const val EXTRA_PATH = "scan_path"
        const val EXTRA_MODE = "scan_mode"
        private const val TAG = "FileScan"
    }
}
