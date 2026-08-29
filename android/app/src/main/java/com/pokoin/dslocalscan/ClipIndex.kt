package com.pokoin.dslocalscan

import android.content.Context
import android.util.Log
import org.json.JSONObject
import org.opencv.core.Mat
import java.io.File

/**
 * CLIP identify v2: crop → 512-d L2 → CardTrader blueprint_id.
 *
 * Encoder (OpenCLIP ViT-B-32) is not in this APK yet. When
 * `files/clip_index/embeddings.bin` + `metadata.jsonl` are present, status
 * is [STATUS_INDEX_NO_ENCODER]. Cosine 1-NN is ready the day the encoder ships.
 * Do not mix these vectors with native HOG 268-d.
 */
object ClipIndex {
    private const val TAG = "ClipIndex"
    const val ASSET_DIR = "clip_index"
    const val STATUS_NO_INDEX = "no_index"
    const val STATUS_INDEX_NO_ENCODER = "index_no_encoder"
    const val STATUS_READY = "ready"

    data class Hit(
        val blueprintId: String,
        val name: String?,
        val setName: String?,
        val version: String?,
        val cosine: Float,
    )

    @Volatile
    var status: String = STATUS_NO_INDEX
        private set

    @Volatile
    var rows: Int = 0
        private set

    fun refresh(context: Context) {
        val dir = File(context.filesDir, ASSET_DIR)
        val npy = File(dir, "embeddings.bin")
        val meta = File(dir, "metadata.jsonl")
        val assetMeta = try {
            context.assets.open("$ASSET_DIR/metadata.jsonl").use { it.readBytes().isNotEmpty() }
        } catch (_: Throwable) {
            false
        }
        status = when {
            npy.isFile && meta.isFile && npy.length() > 0L -> STATUS_INDEX_NO_ENCODER
            assetMeta -> STATUS_INDEX_NO_ENCODER
            else -> STATUS_NO_INDEX
        }
        rows = if (meta.isFile) meta.readLines().count { it.isNotBlank() } else 0
        Log.i(TAG, "clip status=$status rows=$rows")
    }

    /** Live encode is not on-device yet. Always null until STATUS_READY. */
    @Suppress("UNUSED_PARAMETER")
    fun match(warpedBgr: Mat): Hit? = null

    fun infoJson(): JSONObject = JSONObject()
        .put("status", status)
        .put("rows", rows)
        .put("model", "OpenCLIP ViT-B-32 laion2b_s34b_b79k")
        .put("identity", "blueprint_id")
        .put("dim", 512)
}
