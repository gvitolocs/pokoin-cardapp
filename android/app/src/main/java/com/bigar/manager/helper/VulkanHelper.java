package com.bigar.manager.helper;

import android.graphics.Bitmap;
import android.util.Log;

/**
 * JNI bridge for {@code libvulkan_compute.so} (GPU L2 nearest-neighbour + NV21/Bitmap conversion).
 *
 * <p>Singleton shaped like the original Kotlin {@code object VulkanHelper}; the five natives are
 * pinned 1:1 to the {@code Java_com_bigar_manager_helper_VulkanHelper_*} exports of the bundled lib.
 */
public final class VulkanHelper {

    private static final String TAG = "BigarNative";
    public static final VulkanHelper INSTANCE = new VulkanHelper();

    private volatile boolean isInitialized = false;
    private static volatile boolean libraryLoaded = false;

    static {
        try {
            System.loadLibrary("vulkan_compute");
            libraryLoaded = true;
        } catch (UnsatisfiedLinkError e) {
            Log.w(TAG, "vulkan_compute not loaded on this device: " + e.getMessage());
        }
    }

    private VulkanHelper() {
    }

    public boolean isLibraryLoaded() {
        return libraryLoaded;
    }

    public boolean isInitialized() {
        return isInitialized;
    }

    public synchronized boolean initialize() {
        if (isInitialized) {
            return true;
        }
        if (!libraryLoaded) {
            return false;
        }
        try {
            boolean ok = nativeInitVulkan();
            if (ok) {
                isInitialized = true;
            }
            return ok;
        } catch (Throwable t) {
            Log.e(TAG, "Vulkan init failed", t);
            return false;
        }
    }

    public synchronized void cleanup() {
        if (!isInitialized || !libraryLoaded) {
            isInitialized = false;
            return;
        }
        try {
            nativeCleanupVulkan();
        } catch (Throwable t) {
            Log.e(TAG, "Vulkan cleanup failed", t);
        }
        isInitialized = false;
    }

    /** GPU L2 nearest-neighbour: rows(matA) x cols vs 1 query row. Returns best row index or -1. */
    public int computeMatrixDistance(float[] matA, float[] matB, int rows, int cols, float threshold) {
        if (!isInitialized() || !libraryLoaded) {
            throw new IllegalStateException("Vulkan not initialized");
        }
        return nativeComputeMatrixDistance(matA, matB, rows, cols, threshold);
    }

    public Bitmap nv21ToBitmap(byte[] nv21Data, int width, int height) {
        if (!isInitialized()) {
            return null;
        }
        try {
            Bitmap out = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
            if (nativeNv21ToBitmap(nv21Data, width, height, out)) {
                return out;
            }
            out.recycle();
            return null;
        } catch (Throwable t) {
            Log.e(TAG, "nv21ToBitmap failed", t);
            return null;
        }
    }

    public boolean bitmapToNv21(Bitmap bitmap, byte[] outNv21Data, int width, int height) {
        if (!isInitialized() || !libraryLoaded) {
            return false;
        }
        try {
            return nativeBitmapToNv21(bitmap, outNv21Data, width, height);
        } catch (Throwable t) {
            Log.e(TAG, "bitmapToNv21 failed", t);
            return false;
        }
    }

    private final native boolean nativeInitVulkan();

    private final native void nativeCleanupVulkan();

    private final native int nativeComputeMatrixDistance(float[] matA, float[] matB, int rows, int cols, float threshold);

    private final native boolean nativeNv21ToBitmap(byte[] nv21Data, int width, int height, Bitmap outBitmap);

    private final native boolean nativeBitmapToNv21(Bitmap bitmap, byte[] outNv21Data, int width, int height);
}
