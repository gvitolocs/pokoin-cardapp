package com.bigar.manager;

/**
 * JNI bridge for {@code libopencvNative.so} (Pokoin's bundled card engine).
 *
 * <p>Method names and signatures are pinned 1:1 to the native exports
 * ({@code Java_com_bigar_manager_BigarUtils_*}); keep parameter lists aligned.
 *
 * <p>This class deliberately has NO {@code static { loadLibrary(...) }} —
 * the loader lives in {@code com.pokoin.dslocalscan.BigarNative}, which can
 * feature-detect the device ABI and fail gracefully (e.g. x86_64 emulator).
 */
public final class BigarUtils {

    private BigarUtils() {
    }

    /** Load / point the engine at an on-device gallery blob file (the {@code .bigar} we ship as an asset). */
    public static native void nativeExpansionFileManager(String path);

    /**
     * Auto-detect a card, warp it, and fill {@code outHog} (length {@link #getHogs()}).
     *
     * <p>Dragon Shield passes an RGBA {@code Mat} from {@code Utils.bitmapToMat} at CameraX
     * 4:3 sensor size. Native rotates 90° CW, resizes to 288×384, then HOGs. Do not
     * Java-resize to 288×384 first.
     *
     * @param image  native address of an OpenCV RGBA {@code Mat} ({@code mat.nativeObj})
     * @param mode   0 = full-frame rectangle detection
     * @return 1 when a card was found, otherwise 0
     */
    public static native int rectangleDetection(long image, int mode, float[] outHog);

    /** Same as {@link #rectangleDetection} for manual/seeded detections. Returns &gt;0 on hit. */
    public static native int manualDetection(long image, int mode, float[] outHog);

    /** Native address of the last warped card image (engine-internal Mat). */
    public static native long getWarpedCardImage();

    /** Number of gallery rows currently loaded into the engine. */
    public static native int getTotal();

    /** Descriptor dimension (floats per gallery row / per query vector). */
    public static native int getHogs();

    /** Fill {@code values} (length {@code getTotal() * getHogs()}) with the full gallery matrix. */
    public static native void getValues(float[] values);

    /** Map a gallery row index to the layout/card id. */
    public static native int getMinReturnID(int index);

    /** CPU capability fingerprint used for telemetry in the original apps. */
    public static native int getCpuCapabilities();
}
