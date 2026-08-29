package com.pokoin.dslocalscan

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Path
import android.util.AttributeSet
import android.view.View

class QuadOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
) : View(context, attrs) {

    private val path = Path()
    private val stroke = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 5f
        color = 0xFF00E676.toInt()
    }
    private val fill = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = 0x3300E676
    }
    private val missStroke = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 4f
        color = 0xFFFFC107.toInt()
    }
    private val missFill = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = 0x33FFC107
    }

    data class OverlayQuad(val xy: FloatArray, val matched: Boolean)

    @Volatile
    private var quads: List<OverlayQuad> = emptyList()
    @Volatile
    private var imgW = 1
    @Volatile
    private var imgH = 1
    @Volatile
    var matched: Boolean = false
        set(value) {
            field = value
            stroke.color = if (value) 0xFF00E676.toInt() else 0xFFFFC107.toInt()
            fill.color = if (value) 0x3300E676 else 0x33FFC107
            postInvalidateOnAnimation()
        }

    fun setQuad(quad: FloatArray?, width: Int, height: Int) {
        setQuads(if (quad != null && quad.size >= 8) listOf(OverlayQuad(quad, matched)) else emptyList(), width, height)
    }

    fun setQuads(next: List<OverlayQuad>, width: Int, height: Int) {
        quads = next
        if (width > 0) imgW = width
        if (height > 0) imgH = height
        postInvalidateOnAnimation()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (quads.isEmpty()) return
        val sx = width.toFloat() / imgW
        val sy = height.toFloat() / imgH
        for (q in quads) {
            if (q.xy.size < 8) continue
            path.reset()
            path.moveTo(q.xy[0] * sx, q.xy[1] * sy)
            path.lineTo(q.xy[2] * sx, q.xy[3] * sy)
            path.lineTo(q.xy[4] * sx, q.xy[5] * sy)
            path.lineTo(q.xy[6] * sx, q.xy[7] * sy)
            path.close()
            if (q.matched) {
                canvas.drawPath(path, fill)
                canvas.drawPath(path, stroke)
            } else {
                canvas.drawPath(path, missFill)
                canvas.drawPath(path, missStroke)
            }
        }
    }
}
