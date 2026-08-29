package com.pokoin.dslocalscan

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

data class IndexedCard(
    val id: String,
    val name: String,
    val hog: FloatArray,
)

class GalleryStore(context: Context) {
    private val dir = File(context.filesDir, "index").also { it.mkdirs() }
    private val metaFile = File(dir, "cards.json")

    @Volatile
    var cards: List<IndexedCard> = emptyList()
        private set

    fun load() {
        if (!metaFile.exists()) {
            cards = emptyList()
            return
        }
        val arr = JSONArray(metaFile.readText())
        val loaded = ArrayList<IndexedCard>(arr.length())
        for (i in 0 until arr.length()) {
            val o = arr.getJSONObject(i)
            val id = o.getString("id")
            val hogFile = File(dir, "$id.hog")
            if (!hogFile.exists()) continue
            loaded += IndexedCard(id = id, name = o.getString("name"), hog = readHog(hogFile))
        }
        cards = loaded
    }

    fun add(name: String, hog: FloatArray): IndexedCard {
        val id = System.currentTimeMillis().toString(16)
        writeHog(File(dir, "$id.hog"), hog)
        val card = IndexedCard(id, name.ifBlank { "card_${cards.size + 1}" }, hog)
        cards = cards + card
        persistMeta()
        return card
    }

    fun clear() {
        dir.listFiles()?.forEach { it.delete() }
        cards = emptyList()
    }

    fun hogGallery(): List<FloatArray> = cards.map { it.hog }

    private fun persistMeta() {
        val arr = JSONArray()
        for (c in cards) {
            arr.put(JSONObject().put("id", c.id).put("name", c.name))
        }
        metaFile.writeText(arr.toString())
    }

    private fun writeHog(file: File, hog: FloatArray) {
        DataOutputStream(file.outputStream().buffered()).use { out ->
            out.writeInt(hog.size)
            val buf = ByteBuffer.allocate(hog.size * 4).order(ByteOrder.LITTLE_ENDIAN)
            buf.asFloatBuffer().put(hog)
            out.write(buf.array())
        }
    }

    private fun readHog(file: File): FloatArray {
        DataInputStream(file.inputStream().buffered()).use { inp ->
            val n = inp.readInt()
            val raw = ByteArray(n * 4)
            inp.readFully(raw)
            val buf = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
            val hog = FloatArray(n)
            buf.get(hog)
            return hog
        }
    }
}
