package com.pokoin.dslocalscan

import android.util.Log
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.sqrt

/** Stock-OpenCV port of pipeline.py (not BigAR JNI). */
object CardPipeline {
    const val TAG = "DsLocalScan"
    const val CARD_W = 252
    const val CARD_H = 352
    const val HOG_DIM = 10584
    const val DEFAULT_THRESHOLD = 1.4f
    const val INSET_FRAC = 0.06
    const val MIN_AREA_FRAC = 0.04
    const val MIN_SIDE_RATIO = 0.35

    data class DetectResult(
        val quad: Array<Point>,
        val warped: Mat,
        val hog: FloatArray,
        val source: String,
    )

    data class Match(
        val index: Int,
        val distance: Float,
    )

    data class QuadHit(val quad: Array<Point>, val source: String)

    fun detectCardQuad(bgr: Mat): Array<Point>? = detectCardQuadDetailed(bgr)?.quad

    /**
     * Bulk detect: all plausible card quads, largest first, NMS on AABB IoU.
     * Same Canny then adaptive passes as [detectCardQuadDetailed], but keeps
     * more than one 4-gon so a spread of cards can be identified independently.
     */
    fun detectCardQuads(
        bgr: Mat,
        maxHits: Int = 8,
        minAreaFrac: Double = 0.015,
        iouThresh: Double = 0.35,
    ): List<QuadHit> {
        val gray = Mat()
        val edges = Mat()
        val kernel3 = Mat.ones(3, 3, CvType.CV_8U)
        val kernel5 = Mat.ones(5, 5, CvType.CV_8U)
        val hierarchy = Mat()
        val contours = ArrayList<MatOfPoint>()
        val thr = Mat()
        val found = ArrayList<QuadHit>()
        try {
            Imgproc.cvtColor(bgr, gray, Imgproc.COLOR_BGR2GRAY)
            Imgproc.GaussianBlur(gray, gray, Size(5.0, 5.0), 0.0)
            val minArea = minAreaFrac * bgr.cols() * bgr.rows()

            Imgproc.Canny(gray, edges, 40.0, 120.0)
            Imgproc.dilate(edges, edges, kernel3)
            Imgproc.findContours(edges, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
            collectQuads(contours, minArea, "canny_4gon", found)

            contours.forEach { it.release() }
            contours.clear()
            Imgproc.adaptiveThreshold(
                gray, thr, 255.0,
                Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 31, 5.0,
            )
            Imgproc.morphologyEx(thr, thr, Imgproc.MORPH_CLOSE, kernel5)
            Imgproc.findContours(thr, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
            collectQuads(contours, minArea, "adaptive_4gon", found)
            return nmsQuads(found, iouThresh).take(maxHits)
        } finally {
            gray.release()
            edges.release()
            kernel3.release()
            kernel5.release()
            hierarchy.release()
            thr.release()
            contours.forEach { it.release() }
        }
    }

    fun detectCardQuadDetailed(bgr: Mat): QuadHit? {
        val gray = Mat()
        val edges = Mat()
        val kernel3 = Mat.ones(3, 3, CvType.CV_8U)
        val kernel5 = Mat.ones(5, 5, CvType.CV_8U)
        val hierarchy = Mat()
        val contours = ArrayList<MatOfPoint>()
        val thr = Mat()
        try {
            Imgproc.cvtColor(bgr, gray, Imgproc.COLOR_BGR2GRAY)
            Imgproc.GaussianBlur(gray, gray, Size(5.0, 5.0), 0.0)
            val minArea = MIN_AREA_FRAC * bgr.cols() * bgr.rows()

            Imgproc.Canny(gray, edges, 40.0, 120.0)
            Imgproc.dilate(edges, edges, kernel3)
            Imgproc.findContours(edges, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
            val cannyN = contours.size
            val cannyArea = contours.maxOfOrNull { Imgproc.contourArea(it) } ?: 0.0
            quadFromContours(contours, minArea)?.let {
                return QuadHit(insetQuad(it, INSET_FRAC), "canny_4gon")
            }
            minAreaRectQuad(contours, minArea)?.let {
                return QuadHit(insetQuad(it, INSET_FRAC), "canny_min_rect")
            }

            contours.forEach { it.release() }
            contours.clear()
            Imgproc.adaptiveThreshold(
                gray, thr, 255.0,
                Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 31, 5.0,
            )
            Imgproc.morphologyEx(thr, thr, Imgproc.MORPH_CLOSE, kernel5)
            Imgproc.findContours(thr, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
            val adaptN = contours.size
            val adaptArea = contours.maxOfOrNull { Imgproc.contourArea(it) } ?: 0.0
            quadFromContours(contours, minArea)?.let {
                return QuadHit(insetQuad(it, INSET_FRAC), "adaptive_4gon")
            }
            minAreaRectQuad(contours, minArea)?.let {
                return QuadHit(insetQuad(it, INSET_FRAC), "adaptive_min_rect")
            }
            Log.d(
                TAG,
                "detect miss ${bgr.cols()}x${bgr.rows()} minArea=${minArea.toInt()} " +
                    "cannyContours=$cannyN cannyMaxArea=${cannyArea.toInt()} " +
                    "adaptContours=$adaptN adaptMaxArea=${adaptArea.toInt()}",
            )
            return null
        } finally {
            gray.release()
            edges.release()
            kernel3.release()
            kernel5.release()
            hierarchy.release()
            thr.release()
            contours.forEach { it.release() }
        }
    }

    fun warpCard(bgr: Mat, quad: Array<Point>, size: Size = Size(CARD_W.toDouble(), CARD_H.toDouble())): Mat {
        val src = MatOfPoint2f(*quad)
        val dst = MatOfPoint2f(
            Point(0.0, 0.0),
            Point(size.width - 1, 0.0),
            Point(size.width - 1, size.height - 1),
            Point(0.0, size.height - 1),
        )
        val m = Imgproc.getPerspectiveTransform(src, dst)
        val warped = Mat()
        Imgproc.warpPerspective(bgr, warped, m, size)
        m.release()
        src.release()
        dst.release()
        return warped
    }

    fun hogVector(bgr: Mat): FloatArray {
        val gray = Mat()
        val resized = Mat()
        try {
            Imgproc.cvtColor(bgr, gray, Imgproc.COLOR_BGR2GRAY)
            Imgproc.resize(gray, resized, Size(CARD_W.toDouble(), CARD_H.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
            return hogNumpy(resized)
        } finally {
            gray.release()
            resized.release()
        }
    }

    fun describe(bgr: Mat): DetectResult? {
        val hit = detectCardQuadDetailed(bgr) ?: return null
        val warped = warpCard(bgr, hit.quad)
        val hog = hogVector(warped)
        if (hog.size != HOG_DIM) {
            Log.w(TAG, "hog dim ${hog.size} expected $HOG_DIM source=${hit.source}")
        }
        return DetectResult(quad = hit.quad, warped = warped, hog = hog, source = hit.source)
    }

    fun describeFlat(bgr: Mat): FloatArray {
        val warped = Mat()
        try {
            Imgproc.resize(bgr, warped, Size(CARD_W.toDouble(), CARD_H.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
            return hogVector(warped)
        } finally {
            warped.release()
        }
    }

    fun indexImage(bgr: Mat): Pair<FloatArray, Mat> {
        val det = describe(bgr)
        if (det != null) {
            Log.i(TAG, "index via ${det.source} hog=${det.hog.size}")
            return det.hog to det.warped
        }
        Log.w(TAG, "index fallback: no quad, resize ${bgr.cols()}x${bgr.rows()} -> ${CARD_W}x${CARD_H}")
        val warped = Mat()
        Imgproc.resize(bgr, warped, Size(CARD_W.toDouble(), CARD_H.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
        return hogVector(warped) to warped
    }

    fun nearest(query: FloatArray, gallery: List<FloatArray>, threshold: Float = DEFAULT_THRESHOLD): Match {
        if (gallery.isEmpty()) return Match(-1, Float.POSITIVE_INFINITY)
        var bestI = -1
        var bestD = Float.POSITIVE_INFINITY
        var skippedDim = 0
        for (i in gallery.indices) {
            val g = gallery[i]
            if (g.size != query.size) {
                skippedDim++
                continue
            }
            var s = 0f
            for (j in query.indices) {
                val d = g[j] - query[j]
                s += d * d
            }
            val d = sqrt(s)
            if (d < bestD) {
                bestD = d
                bestI = i
            }
        }
        if (skippedDim > 0) {
            Log.w(TAG, "nearest skipped $skippedDim gallery vecs (query dim=${query.size})")
        }
        return if (bestD > threshold) Match(-1, bestD) else Match(bestI, bestD)
    }

    private fun minAreaRectQuad(contours: List<MatOfPoint>, minArea: Double): Array<Point>? {
        if (contours.isEmpty()) return null
        val largest = contours.maxBy { Imgproc.contourArea(it) }
        if (Imgproc.contourArea(largest) < minArea) return null
        val pts2f = MatOfPoint2f(*largest.toArray())
        val rect = Imgproc.minAreaRect(pts2f)
        pts2f.release()
        val box = arrayOf(Point(), Point(), Point(), Point())
        rect.points(box)
        val ordered = orderQuad(box)
        return if (plausibleCardQuad(ordered)) ordered else null
    }

    private fun collectQuads(
        contours: List<MatOfPoint>,
        minArea: Double,
        source: String,
        out: MutableList<QuadHit>,
    ) {
        for (c in contours) {
            val c2f = MatOfPoint2f(*c.toArray())
            val peri = Imgproc.arcLength(c2f, true)
            val approx = MatOfPoint2f()
            Imgproc.approxPolyDP(c2f, approx, 0.02 * peri, true)
            c2f.release()
            val pts = approx.toArray()
            val approxPts = MatOfPoint(*pts)
            val convex = pts.size == 4 && Imgproc.isContourConvex(approxPts)
            val area = Imgproc.contourArea(approx)
            approx.release()
            approxPts.release()
            if (!convex || area < minArea) continue
            val ordered = orderQuad(pts)
            if (!plausibleCardQuad(ordered)) continue
            out.add(QuadHit(insetQuad(ordered, INSET_FRAC), source))
        }
    }

    private fun nmsQuads(hits: List<QuadHit>, iouThresh: Double): List<QuadHit> {
        val scored = hits.map { it to aabbArea(it.quad) }.sortedByDescending { it.second }
        val kept = ArrayList<QuadHit>()
        for ((hit, _) in scored) {
            if (kept.none { aabbIou(it.quad, hit.quad) > iouThresh }) {
                kept.add(hit)
            }
        }
        return kept
    }

    private fun aabb(quad: Array<Point>): DoubleArray {
        val xs = quad.map { it.x }
        val ys = quad.map { it.y }
        return doubleArrayOf(xs.minOrNull()!!, ys.minOrNull()!!, xs.maxOrNull()!!, ys.maxOrNull()!!)
    }

    private fun aabbArea(quad: Array<Point>): Double {
        val b = aabb(quad)
        return (b[2] - b[0]).coerceAtLeast(0.0) * (b[3] - b[1]).coerceAtLeast(0.0)
    }

    private fun aabbIou(a: Array<Point>, b: Array<Point>): Double {
        val A = aabb(a)
        val B = aabb(b)
        val ix = maxOf(0.0, minOf(A[2], B[2]) - maxOf(A[0], B[0]))
        val iy = maxOf(0.0, minOf(A[3], B[3]) - maxOf(A[1], B[1]))
        val inter = ix * iy
        val union = aabbArea(a) + aabbArea(b) - inter
        return if (union <= 0.0) 0.0 else inter / union
    }

    private fun quadFromContours(contours: List<MatOfPoint>, minArea: Double): Array<Point>? {
        var best: Array<Point>? = null
        var bestArea = 0.0
        for (c in contours) {
            val c2f = MatOfPoint2f(*c.toArray())
            val peri = Imgproc.arcLength(c2f, true)
            val approx = MatOfPoint2f()
            Imgproc.approxPolyDP(c2f, approx, 0.02 * peri, true)
            c2f.release()
            val pts = approx.toArray()
            val approxPts = MatOfPoint(*pts)
            val convex = pts.size == 4 && Imgproc.isContourConvex(approxPts)
            val area = Imgproc.contourArea(approx)
            approx.release()
            approxPts.release()
            if (!convex || area < minArea || area <= bestArea) continue
            val ordered = orderQuad(pts)
            if (!plausibleCardQuad(ordered)) continue
            best = ordered
            bestArea = area
        }
        return best
    }

    private fun orderQuad(pts: Array<Point>): Array<Point> {
        val s = pts.map { it.x + it.y }
        val diff = pts.map { it.y - it.x }
        val tl = pts[s.indices.minBy { s[it] }]
        val br = pts[s.indices.maxBy { s[it] }]
        val tr = pts[diff.indices.minBy { diff[it] }]
        val bl = pts[diff.indices.maxBy { diff[it] }]
        return arrayOf(tl, tr, br, bl)
    }

    private fun insetQuad(quad: Array<Point>, frac: Double): Array<Point> {
        val cx = quad.map { it.x }.average()
        val cy = quad.map { it.y }.average()
        return Array(4) { i ->
            Point(quad[i].x * (1.0 - frac) + cx * frac, quad[i].y * (1.0 - frac) + cy * frac)
        }
    }

    private fun plausibleCardQuad(quad: Array<Point>): Boolean {
        val d = DoubleArray(4) { i ->
            val a = quad[i]
            val b = quad[(i + 1) % 4]
            val dx = a.x - b.x
            val dy = a.y - b.y
            sqrt(dx * dx + dy * dy)
        }
        if (d.minOrNull()!! < 8.0) return false
        val w = 0.5 * (d[0] + d[2])
        val h = 0.5 * (d[1] + d[3])
        return (minOf(w, h) / maxOf(w, h)) >= MIN_SIDE_RATIO
    }

    /** Same unsigned 9-bin 16px-cell HOG as pipeline._hog_numpy. */
    private fun hogNumpy(gray: Mat, cell: Int = 16, nbins: Int = 9): FloatArray {
        val gx = Mat()
        val gy = Mat()
        val mag = Mat()
        val ang = Mat()
        try {
            Imgproc.Sobel(gray, gx, CvType.CV_32F, 1, 0, 1)
            Imgproc.Sobel(gray, gy, CvType.CV_32F, 0, 1, 1)
            Core.cartToPolar(gx, gy, mag, ang, true)
            val cellsY = gray.rows() / cell
            val cellsX = gray.cols() / cell
            val croppedH = cellsY * cell
            val croppedW = cellsX * cell
            val magCrop = mag.submat(0, croppedH, 0, croppedW)
            val angCrop = ang.submat(0, croppedH, 0, croppedW)
            val magBuf = FloatArray(croppedH * croppedW)
            val angBuf = FloatArray(croppedH * croppedW)
            magCrop.get(0, 0, magBuf)
            angCrop.get(0, 0, angBuf)
            magCrop.release()
            angCrop.release()

            val binW = 180f / nbins
            val hist = FloatArray(cellsY * cellsX * nbins)
            for (by in 0 until cellsY) {
                for (bx in 0 until cellsX) {
                    val cellBase = (by * cellsX + bx) * nbins
                    for (cy in 0 until cell) {
                        for (cx in 0 until cell) {
                            val idx = (by * cell + cy) * croppedW + (bx * cell + cx)
                            var unsigned = angBuf[idx] % 180f
                            if (unsigned < 0f) unsigned += 180f
                            val b0 = (kotlin.math.floor(unsigned / binW).toInt() % nbins)
                            hist[cellBase + b0] += magBuf[idx]
                        }
                    }
                }
            }

            val blocksY = cellsY - 1
            val blocksX = cellsX - 1
            val blockDim = 4 * nbins
            val out = FloatArray(blocksY * blocksX * blockDim)
            var o = 0
            val block = FloatArray(blockDim)
            for (by in 0 until blocksY) {
                for (bx in 0 until blocksX) {
                    var bi = 0
                    for (dy in 0..1) {
                        for (dx in 0..1) {
                            val cellIdx = ((by + dy) * cellsX + (bx + dx)) * nbins
                            for (b in 0 until nbins) {
                                block[bi++] = hist[cellIdx + b]
                            }
                        }
                    }
                    var n = 1e-6f
                    for (v in block) n += v * v
                    n = sqrt(n)
                    for (v in block) out[o++] = v / n
                }
            }
            var n = 0f
            for (v in out) n += v * v
            n = sqrt(n)
            if (n > 0f) {
                for (i in out.indices) out[i] /= n
            }
            return out
        } finally {
            gx.release()
            gy.release()
            mag.release()
            ang.release()
        }
    }
}
