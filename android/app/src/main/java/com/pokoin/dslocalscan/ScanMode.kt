package com.pokoin.dslocalscan

import android.content.Context

/**
 * Two implementations of the same frame-to-hits contract.
 *
 * Desktop (measured) and this APK Fast/Multi:
 *   FAST: TCG YOLO detect + Milo 128-d ONNX → TCGplayer. No OCR.
 *   MULTI: every YOLO box + Milo. CLIP ViT → blueprint_id when the encoder
 *   ships. RapidOCR is not on the live path.
 * HOG 268-d → LAYOUT is fallback only if YOLO/Milo fail to load.
 * Never mix 268-d HOG, 128-d Milo, and 512-d CLIP as one identity.
 */
enum class ScanMode {
    FAST,
    MULTI,
    ;

    companion object {
        private const val PREFS = "pokoin_scan"
        private const val KEY = "mode"

        fun fromExtra(raw: String?): ScanMode =
            if (raw.equals("multi", ignoreCase = true)) MULTI else FAST

        fun load(context: Context): ScanMode =
            fromExtra(context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getString(KEY, "fast"))

        fun save(context: Context, mode: ScanMode) {
            context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
                .edit()
                .putString(KEY, if (mode == MULTI) "multi" else "fast")
                .apply()
        }
    }
}
