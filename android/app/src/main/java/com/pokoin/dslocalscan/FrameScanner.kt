package com.pokoin.dslocalscan

import org.json.JSONArray
import org.json.JSONObject
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

/**
 * One frame → hits. Gallery is not modified.
 *
 * FAST: TCG YOLO TFLite + Milo 128-d ONNX → TCGplayer. HOG LAYOUT only if
 * YOLO/Milo fail to load. MULTI: every YOLO box + Milo; CLIP encoder is
 * still not packaged. Never mix 268-d HOG, 128-d Milo, 512-d CLIP.
 */
object FrameScanner {
    data class Hit(
        val layoutId: Int,
        val l2: Float,
        val accepted: Boolean,
        val detectRc: Int,
        val hogMs: Long,
        val hogSource: String?,
        val blueprintId: String?,
        val clipCosine: Float?,
        val clipStatus: String,
        val quadSource: String?,
        val quad: FloatArray?,
        val name: String? = null,
        val tcgplayerId: String? = null,
        val collectorNumber: String? = null,
        val miloScore: Float? = null,
        val miloMs: Long = -1,
        val yoloConf: Float? = null,
    )

    data class Result(
        val mode: ScanMode,
        val imgW: Int,
        val imgH: Int,
        val hits: List<Hit>,
        val nativeMs: Long,
        val totalMs: Long,
        val lastWarp: Mat? = null,
    ) {
        val matched: Boolean get() = hits.any {
            it.accepted || it.layoutId > 0 || !it.name.isNullOrBlank()
        }
        val quads: List<FloatArray> get() = hits.mapNotNull { it.quad }
    }

    fun init(context: android.content.Context) {
        YoloDetector.init(context)
        MiloEmbedder.init(context)
        MiloIndex.init(context)
        ClipIndex.refresh(context)
    }

    fun scanStill(displayRgba: Mat, mode: ScanMode): Result {
        val started = System.currentTimeMillis()
        val bgr = Mat()
        Imgproc.cvtColor(displayRgba, bgr, Imgproc.COLOR_RGBA2BGR)
        val result = try {
            when (mode) {
                ScanMode.FAST -> scanFast(displayRgba, bgr, live = false)
                ScanMode.MULTI -> scanMulti(bgr)
            }
        } finally {
            bgr.release()
        }
        return result.copy(totalMs = System.currentTimeMillis() - started)
    }

    fun scanLive(sensorRgba: Mat, rotationDegrees: Int, mode: ScanMode): Result {
        val started = System.currentTimeMillis()
        val displayBgr = MainActivity.rgbaToDisplayBgr(sensorRgba, rotationDegrees)
        val result = try {
            when (mode) {
                ScanMode.FAST -> scanFast(sensorRgba, displayBgr, live = true)
                ScanMode.MULTI -> scanMulti(displayBgr)
            }
        } finally {
            displayBgr.release()
        }
        return result.copy(totalMs = System.currentTimeMillis() - started)
    }

    private fun keeperReady(): Boolean =
        YoloDetector.available && MiloEmbedder.available && MiloIndex.available

    private fun scanFast(nativeRgba: Mat, displayBgr: Mat, live: Boolean): Result {
        if (keeperReady()) {
            val boxes = YoloDetector.detect(displayBgr)
            if (boxes.isNotEmpty()) {
                return miloHits(ScanMode.FAST, displayBgr, listOf(boxes.first()))
            }
        }
        val native = if (live) {
            BigarNative.scan(nativeRgba, requireAccept = false)
        } else {
            BigarNative.scanDisplayRgba(nativeRgba, requireAccept = false)
        }
        return finishHogFast(native, displayBgr)
    }

    private fun scanMulti(bgr: Mat): Result {
        if (keeperReady()) {
            val boxes = YoloDetector.detect(bgr)
            if (boxes.isNotEmpty()) {
                return miloHits(ScanMode.MULTI, bgr, boxes.take(10))
            }
        }
        return scanMultiHog(bgr)
    }

    private fun miloHits(mode: ScanMode, bgr: Mat, boxes: List<YoloDetector.Box>): Result {
        val hits = ArrayList<Hit>(boxes.size)
        var firstWarp: Mat? = null
        var miloMsTotal = 0L
        for (box in boxes) {
            val crop = cropBox(bgr, box)
            val t0 = System.currentTimeMillis()
            val emb = MiloEmbedder.embed(crop)
            val miloMs = System.currentTimeMillis() - t0
            miloMsTotal += miloMs
            val top = emb?.let { MiloIndex.search(it, 5).firstOrNull() }
            val clip = if (mode == ScanMode.MULTI) ClipIndex.match(crop) else null
            hits.add(
                Hit(
                    layoutId = -1,
                    l2 = Float.MAX_VALUE,
                    accepted = top != null && top.score >= 0.45f,
                    detectRc = 1,
                    hogMs = -1,
                    hogSource = null,
                    blueprintId = clip?.blueprintId,
                    clipCosine = clip?.cosine,
                    clipStatus = if (mode == ScanMode.FAST) "skipped_fast" else ClipIndex.status,
                    quadSource = "yolo",
                    quad = box.toQuad(),
                    name = top?.card?.name,
                    tcgplayerId = top?.card?.id,
                    collectorNumber = top?.card?.collectorNumber,
                    miloScore = top?.score,
                    miloMs = miloMs,
                    yoloConf = box.conf,
                ),
            )
            if (firstWarp == null) firstWarp = crop else crop.release()
        }
        return Result(
            mode = mode,
            imgW = bgr.cols(),
            imgH = bgr.rows(),
            hits = hits,
            nativeMs = miloMsTotal,
            totalMs = 0,
            lastWarp = firstWarp,
        )
    }

    private fun cropBox(bgr: Mat, box: YoloDetector.Box): Mat {
        val x1 = box.x1.toInt().coerceIn(0, bgr.cols() - 1)
        val y1 = box.y1.toInt().coerceIn(0, bgr.rows() - 1)
        val x2 = box.x2.toInt().coerceIn(x1 + 1, bgr.cols())
        val y2 = box.y2.toInt().coerceIn(y1 + 1, bgr.rows())
        return bgr.submat(y1, y2, x1, x2).clone()
    }

    private fun finishHogFast(native: BigarNative.ScanResult?, displayBgr: Mat): Result {
        val overlay = CardPipeline.detectCardQuadDetailed(displayBgr)
        val warp = overlay?.quad?.let { CardPipeline.warpCard(displayBgr, it) }
        val hit = Hit(
            layoutId = native?.cardId ?: -1,
            l2 = native?.l2 ?: Float.MAX_VALUE,
            accepted = native?.accepted ?: false,
            detectRc = native?.detectRc ?: 0,
            hogMs = native?.ms ?: -1,
            hogSource = native?.source,
            blueprintId = null,
            clipCosine = null,
            clipStatus = "skipped_fast",
            quadSource = overlay?.source,
            quad = overlay?.quad?.let { toFlat(it) },
        )
        return Result(
            mode = ScanMode.FAST,
            imgW = displayBgr.cols(),
            imgH = displayBgr.rows(),
            hits = listOf(hit),
            nativeMs = native?.ms ?: -1,
            totalMs = 0,
            lastWarp = warp,
        )
    }

    private fun scanMultiHog(bgr: Mat): Result {
        val quads = CardPipeline.detectCardQuads(bgr)
        val hits = ArrayList<Hit>(quads.size)
        var nativeMs = 0L
        var firstWarp: Mat? = null
        for (q in quads) {
            val warped = CardPipeline.warpCard(bgr, q.quad, Size(480.0, 640.0))
            val cropRgba = Mat()
            Imgproc.cvtColor(warped, cropRgba, Imgproc.COLOR_BGR2RGBA)
            val hog = BigarNative.scanDisplayRgba(cropRgba, requireAccept = false)
            nativeMs += hog?.ms ?: 0L
            val clip = ClipIndex.match(warped)
            hits.add(
                Hit(
                    layoutId = hog?.cardId ?: -1,
                    l2 = hog?.l2 ?: Float.MAX_VALUE,
                    accepted = hog?.accepted ?: false,
                    detectRc = hog?.detectRc ?: 0,
                    hogMs = hog?.ms ?: -1,
                    hogSource = hog?.source,
                    blueprintId = clip?.blueprintId,
                    clipCosine = clip?.cosine,
                    clipStatus = if (clip != null) ClipIndex.STATUS_READY else ClipIndex.status,
                    quadSource = q.source,
                    quad = toFlat(q.quad),
                ),
            )
            cropRgba.release()
            if (firstWarp == null) firstWarp = warped else warped.release()
        }
        if (hits.isEmpty()) {
            hits.add(
                Hit(
                    layoutId = -1,
                    l2 = Float.MAX_VALUE,
                    accepted = false,
                    detectRc = 0,
                    hogMs = -1,
                    hogSource = null,
                    blueprintId = null,
                    clipCosine = null,
                    clipStatus = ClipIndex.status,
                    quadSource = null,
                    quad = null,
                ),
            )
        }
        return Result(
            mode = ScanMode.MULTI,
            imgW = bgr.cols(),
            imgH = bgr.rows(),
            hits = hits,
            nativeMs = nativeMs,
            totalMs = 0,
            lastWarp = firstWarp,
        )
    }

    fun statusLine(result: Result, indexSize: Int): String {
        val mode = if (result.mode == ScanMode.FAST) "FAST" else "MULTI"
        val cards = result.hits.filter { it.quad != null || it.layoutId > 0 || it.name != null }
        if (cards.isEmpty() || (result.mode == ScanMode.MULTI && result.hits.all { it.quad == null })) {
            return "$mode · no card · ${result.totalMs}ms · milo ${MiloIndex.size}"
        }
        if (result.mode == ScanMode.FAST) {
            val h = result.hits.first()
            if (h.name != null) {
                val n = h.collectorNumber ?: ""
                val sc = h.miloScore?.let { "%.3f".format(it) } ?: "—"
                return "$mode · ${h.name} $n · $sc · ${result.totalMs}ms"
            }
            val id = if (h.layoutId > 0) "LAYOUT ${h.layoutId}" else "no LAYOUT"
            val acc = if (h.accepted) "ok" else "reject"
            return "$mode · $id L2 ${fmt(h.l2)} $acc · ${result.totalMs}ms"
        }
        val parts = result.hits.filter { it.quad != null }.map { h ->
            h.name ?: h.blueprintId ?: h.clipStatus
        }
        return "$mode · ${parts.size} cards · ${parts.joinToString(" · ")} · ${result.totalMs}ms"
    }

    fun toJson(path: String, result: Result): JSONObject {
        val hits = JSONArray()
        for (h in result.hits) {
            hits.put(
                JSONObject()
                    .put("layout_id", h.layoutId)
                    .put("l2", if (h.l2.isFinite() && h.l2 < 1e30f) h.l2.toDouble() else JSONObject.NULL)
                    .put("accepted", h.accepted)
                    .put("detect_rc", h.detectRc)
                    .put("hog_ms", h.hogMs)
                    .put("hog_source", h.hogSource ?: JSONObject.NULL)
                    .put("blueprint_id", h.blueprintId ?: JSONObject.NULL)
                    .put("clip_cosine", h.clipCosine?.toDouble() ?: JSONObject.NULL)
                    .put("clip_status", h.clipStatus)
                    .put("quad_source", h.quadSource ?: JSONObject.NULL)
                    .put("name", h.name ?: JSONObject.NULL)
                    .put("tcgplayer_id", h.tcgplayerId ?: JSONObject.NULL)
                    .put("collector_number", h.collectorNumber ?: JSONObject.NULL)
                    .put("milo_score", h.miloScore?.toDouble() ?: JSONObject.NULL)
                    .put("milo_ms", h.miloMs)
                    .put("yolo_conf", h.yoloConf?.toDouble() ?: JSONObject.NULL),
            )
        }
        val first = result.hits.firstOrNull()
        val firstL2 = first?.l2
        return JSONObject()
            .put("path", path)
            .put("ok", result.hits.any { it.accepted || it.layoutId > 0 || !it.name.isNullOrBlank() })
            .put("mode", if (result.mode == ScanMode.FAST) "fast" else "multi")
            .put("img_w", result.imgW)
            .put("img_h", result.imgH)
            .put("quads", result.hits.count { it.quad != null })
            .put("native_ms", result.nativeMs)
            .put("total_ms", result.totalMs)
            .put("detect_rc", first?.detectRc ?: 0)
            .put("layout_id", first?.layoutId ?: -1)
            .put(
                "l2",
                if (firstL2 != null && firstL2.isFinite() && firstL2 < 1e30f) {
                    firstL2.toDouble()
                } else {
                    JSONObject.NULL
                },
            )
            .put("accepted", first?.accepted ?: false)
            .put("ms", result.totalMs)
            .put("clip", ClipIndex.infoJson())
            .put("yolo", YoloDetector.available)
            .put("milo", MiloEmbedder.available && MiloIndex.available)
            .put("milo_n", MiloIndex.size)
            .put("hits", hits)
    }

    private fun toFlat(quad: Array<org.opencv.core.Point>): FloatArray {
        val out = FloatArray(8)
        quad.forEachIndexed { i, p ->
            out[i * 2] = p.x.toFloat()
            out[i * 2 + 1] = p.y.toFloat()
        }
        return out
    }

    private fun fmt(v: Float): String =
        if (!v.isFinite() || v > 1e20f) "—" else "%.3f".format(v)
}
