package com.pokoin.dslocalscan

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.view.WindowManager
import android.widget.EditText
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.core.view.doOnLayout
import com.pokoin.dslocalscan.databinding.ActivityMainBinding
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
    private lateinit var gallery: GalleryStore
    private val analysisExecutor = Executors.newSingleThreadExecutor()
    private val busy = AtomicBoolean(false)
    private val warpLock = Any()
    private var lastWarped: Mat? = null
    private var cameraBound = false
    @Volatile private var scanMode = ScanMode.FAST

    private val askCamera = registerForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { granted ->
        if (granted) {
            binding.previewView.doOnLayout { bindCamera() }
        } else {
            toast("Camera permission required")
        }
    }

    private val pickPhoto = registerForActivityResult(
        ActivityResultContracts.GetContent(),
    ) { uri ->
        if (uri != null) indexFromUri(uri)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        if (!OpenCVLoader.initLocal()) {
            toast("OpenCV failed to load")
            finish()
            return
        }

        gallery = GalleryStore(this)
        gallery.load()
        refreshIndexLabel()
        scanMode = ScanMode.load(this)
        binding.modeGroup.check(if (scanMode == ScanMode.MULTI) R.id.btnMulti else R.id.btnFast)
        binding.modeGroup.addOnButtonCheckedListener { _, checkedId, isChecked ->
            if (!isChecked) return@addOnButtonCheckedListener
            scanMode = if (checkedId == R.id.btnMulti) ScanMode.MULTI else ScanMode.FAST
            ScanMode.save(this, scanMode)
            binding.status.text = if (scanMode == ScanMode.MULTI) {
                getString(R.string.mode_hint_multi)
            } else {
                getString(R.string.mode_hint_fast)
            }
        }

        // Boot the bundled native engine (BigarUtils + bundled CDN index) off the camera thread.
        Thread {
            try {
                BigarNative.initGalleryIndex(this)
                BigarNative.refreshAvailability()
                FrameScanner.init(this)
                Log.i(TAG, "bootstrap yolo=${YoloDetector.available} milo=${MiloEmbedder.available} n=${MiloIndex.size} hog=${BigarNative.available} clip=${ClipIndex.status}")
            } catch (t: Throwable) {
                Log.w(TAG, "BigarNative bootstrap failed (continuing with stock pipeline)", t)
            }
        }.start()

        binding.btnAddPhoto.setOnClickListener { pickPhoto.launch("image/*") }
        binding.btnSaveWarp.setOnClickListener { saveLastWarp() }
        binding.btnClear.setOnClickListener {
            gallery.clear()
            refreshIndexLabel()
            toast("Index cleared")
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
            PackageManager.PERMISSION_GRANTED
        ) {
            binding.previewView.doOnLayout { bindCamera() }
        } else {
            askCamera.launch(Manifest.permission.CAMERA)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        analysisExecutor.shutdown()
        synchronized(warpLock) {
            lastWarped?.release()
            lastWarped = null
        }
    }

    private fun bindCamera() {
        if (cameraBound) return
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().build().also {
                it.surfaceProvider = binding.previewView.surfaceProvider
            }
            val analysis = ImageAnalysis.Builder()
                .setResolutionSelector(
                    ResolutionSelector.Builder()
                        .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
                        .build(),
                )
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
            analysis.setAnalyzer(analysisExecutor, ::analyzeFrame)

            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                this,
                CameraSelector.DEFAULT_BACK_CAMERA,
                preview,
                analysis,
            )
            cameraBound = true
        }, ContextCompat.getMainExecutor(this))
    }

    private fun analyzeFrame(image: ImageProxy) {
        if (!busy.compareAndSet(false, true)) {
            image.close()
            return
        }
        try {
            val rotation = image.imageInfo.rotationDegrees
            val rgba = imageProxyToRgba(image)
            val result = try {
                FrameScanner.scanLive(rgba, rotation, scanMode)
            } finally {
                rgba.release()
            }
            val overlayQuads = result.hits.mapNotNull { h ->
                val q = h.quad ?: return@mapNotNull null
                QuadOverlayView.OverlayQuad(q, h.accepted || h.layoutId > 0 || !h.name.isNullOrBlank())
            }
            synchronized(warpLock) {
                lastWarped?.release()
                lastWarped = result.lastWarp
            }
            runOnUiThread {
                binding.overlay.setQuads(overlayQuads, result.imgW, result.imgH)
                binding.overlay.matched = result.matched
                binding.status.text = FrameScanner.statusLine(result, gallery.cards.size)
            }
        } catch (e: Exception) {
            Log.e(TAG, "analyze", e)
        } finally {
            image.close()
            busy.set(false)
        }
    }

    private fun saveLastWarp() {
        if (synchronized(warpLock) { lastWarped == null }) {
            toast("No card in view")
            return
        }
        promptName("ref_${gallery.cards.size + 1}") { name ->
            val copy = synchronized(warpLock) { lastWarped?.clone() }
            if (copy == null) {
                toast("No card in view")
                return@promptName
            }
            analysisExecutor.execute {
                try {
                    val hog = CardPipeline.hogVector(copy)
                    runOnUiThread {
                        gallery.add(name, hog)
                        refreshIndexLabel()
                        toast("Saved $name")
                    }
                } finally {
                    copy.release()
                }
            }
        }
    }

    private fun indexFromUri(uri: Uri) {
        analysisExecutor.execute {
            try {
                val bmp = loadBitmap(uri) ?: run {
                    runOnUiThread { toast("Could not read image") }
                    return@execute
                }
                val rgba = Mat()
                Utils.bitmapToMat(bmp, rgba)
                val bgr = Mat()
                Imgproc.cvtColor(rgba, bgr, Imgproc.COLOR_RGBA2BGR)
                rgba.release()
                bmp.recycle()
                val (hog, warped) = CardPipeline.indexImage(bgr)
                bgr.release()
                warped.release()
                runOnUiThread {
                    promptName(uri.lastPathSegment?.substringBeforeLast('.') ?: "card") { name ->
                        gallery.add(name, hog)
                        refreshIndexLabel()
                        toast("Indexed $name")
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "index", e)
                runOnUiThread { toast("Index failed: ${e.message}") }
            }
        }
    }

    private fun loadBitmap(uri: Uri): Bitmap? {
        val raw = if (Build.VERSION.SDK_INT >= 28) {
            val src = ImageDecoder.createSource(contentResolver, uri)
            ImageDecoder.decodeBitmap(src) { decoder, _, _ ->
                decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
                decoder.isMutableRequired = true
            }
        } else {
            @Suppress("DEPRECATION")
            android.provider.MediaStore.Images.Media.getBitmap(contentResolver, uri)
        }
        return if (raw.config == Bitmap.Config.ARGB_8888) raw else raw.copy(Bitmap.Config.ARGB_8888, true)
    }

    private fun promptName(default: String, onOk: (String) -> Unit) {
        val input = EditText(this).apply {
            setText(default)
            setSelection(default.length)
        }
        AlertDialog.Builder(this)
            .setTitle("Index name")
            .setView(input)
            .setPositiveButton("Save") { _, _ -> onOk(input.text.toString().trim()) }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun refreshIndexLabel() {
        binding.indexCount.text = getString(R.string.index_count, gallery.cards.size)
    }

    private fun toast(msg: String) = Toast.makeText(this, msg, Toast.LENGTH_SHORT).show()

    companion object {
        private const val TAG = "DsLocalScan"

        fun imageProxyToRgba(image: ImageProxy): Mat {
            val plane = image.planes[0]
            val buf = plane.buffer
            val rowStride = plane.rowStride
            val pixelStride = plane.pixelStride
            val h = image.height
            val w = image.width
            val rgba = Mat(h, w, CvType.CV_8UC4)
            buf.rewind()
            if (pixelStride == 4 && rowStride == w * 4) {
                val bytes = ByteArray(buf.remaining())
                buf.get(bytes)
                rgba.put(0, 0, bytes)
            } else {
                val row = ByteArray(rowStride)
                val compact = ByteArray(w * 4)
                for (y in 0 until h) {
                    buf.position(y * rowStride)
                    val n = rowStride.coerceAtMost(buf.remaining())
                    buf.get(row, 0, n)
                    if (pixelStride == 4) {
                        System.arraycopy(row, 0, compact, 0, w * 4)
                    } else {
                        for (x in 0 until w) {
                            val s = x * pixelStride
                            val d = x * 4
                            compact[d] = row[s]
                            compact[d + 1] = row[s + 1]
                            compact[d + 2] = row[s + 2]
                            compact[d + 3] = row[s + 3]
                        }
                    }
                    rgba.put(y, 0, compact)
                }
            }
            return rgba
        }

        fun rgbaToDisplayBgr(rgba: Mat, rotationDegrees: Int): Mat {
            val bgr = Mat()
            Imgproc.cvtColor(rgba, bgr, Imgproc.COLOR_RGBA2BGR)
            val flag = when (rotationDegrees) {
                90 -> Core.ROTATE_90_CLOCKWISE
                180 -> Core.ROTATE_180
                270 -> Core.ROTATE_90_COUNTERCLOCKWISE
                else -> return bgr
            }
            val rotated = Mat()
            Core.rotate(bgr, rotated, flag)
            bgr.release()
            return rotated
        }
    }
}
