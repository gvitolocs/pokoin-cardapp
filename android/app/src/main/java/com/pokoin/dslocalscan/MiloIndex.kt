package com.pokoin.dslocalscan

import android.content.Context
import android.util.Log
import org.json.JSONObject
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Exact cosine 1-NN over the bundled Milo pokemon catalog (TCGplayer ids). */
object MiloIndex {
    private const val TAG = "MiloIndex"
    private const val BIN = "milo_index/embeddings.bin"
    private const val META = "milo_index/metadata.jsonl"

    data class Card(
        val id: String,
        val name: String,
        val collectorNumber: String?,
        val set: String?,
    )

    data class Hit(val card: Card, val score: Float, val index: Int)

    @Volatile var available: Boolean = false
        private set
    @Volatile var size: Int = 0
        private set

    private var vecs: FloatArray = FloatArray(0)
    private var dim: Int = 128
    private var cards: Array<Card> = emptyArray()

    fun init(context: Context) {
        synchronized(this) {
            if (available) return
            try {
            val bytes = context.assets.open(BIN).use { it.readBytes() }
            val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
            val nFloats = buf.remaining()
            if (nFloats % dim != 0) {
                throw IllegalStateException("embeddings.bin length $nFloats not divisible by $dim")
            }
            val n = nFloats / dim
            val copy = FloatArray(nFloats)
            buf.get(copy)
            val recs = ArrayList<Card>(n)
            context.assets.open(META).bufferedReader().useLines { lines ->
                lines.forEach { line ->
                    if (line.isBlank()) return@forEach
                    val o = JSONObject(line)
                    recs.add(
                        Card(
                            id = o.optString("id"),
                            name = o.optString("name"),
                            collectorNumber = o.optString("n").ifEmpty { null },
                            set = o.optString("set").ifEmpty { null },
                        ),
                    )
                }
            }
            if (recs.size != n) {
                throw IllegalStateException("metadata ${recs.size} vs embeddings $n")
            }
            vecs = copy
            cards = recs.toTypedArray()
            size = n
            available = true
            Log.i(TAG, "loaded n=$n dim=$dim")
            } catch (t: Throwable) {
                Log.w(TAG, "Milo index unavailable", t)
                available = false
            }
        }
    }

    fun search(query: FloatArray, topK: Int = 5): List<Hit> {
        if (!available || query.size != dim) return emptyList()
        val scores = FloatArray(size)
        val v = vecs
        val d = dim
        for (i in 0 until size) {
            var s = 0f
            val base = i * d
            for (j in 0 until d) s += v[base + j] * query[j]
            scores[i] = s
        }
        val order = scores.indices.sortedByDescending { scores[it] }.take(topK)
        return order.map { Hit(cards[it], scores[it], it) }
    }
}
