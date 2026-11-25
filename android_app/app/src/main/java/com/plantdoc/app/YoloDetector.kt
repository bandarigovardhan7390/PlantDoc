package com.plantdoc.app

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import org.json.JSONArray
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/** One leaf found in the photo. Coordinates are pixels of the ORIGINAL bitmap that was given to detect(). */
data class Detection(
    val label: String,
    val confidence: Float,
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float
)

/**
 * YOLO26 leaf-disease detector running on ONNX Runtime.
 * Pre-processing is the same as export_yolo26_onnx.py (checked against PyTorch on the PC):
 *   keep the photo's shape, centre it on a 640x640 canvas padded with grey (114), RGB, divide by 255,
 *   channel-first. NO ImageNet mean/std (that was for the old MobileNet).
 * Model output [1, 300, 6] = x1, y1, x2, y2, confidence, classId in 640x640 canvas pixels (no NMS needed).
 *
 * The model was trained on photos where a leaf covers only about 7 to 13% of the picture, so close-ups work badly.
 * detectMultiScale() also looks at smaller copies of the photo (shrunk on the grey canvas) and keeps the best answer.
 */
class YoloDetector(context: Context) {

    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val inputName: String
    val classNames: List<String>

    companion object {
        const val SIZE = 640
        private const val PAD = 114f / 255f
        private const val MODEL_FILE = "leaf_disease_yolo26n.onnx"
        private const val CLASS_FILE = "class_names.json"

        // Extra passes are only made when the first answer is weak or the leaf fills most of the photo.
        private const val CONFIDENT = 0.75f
        private const val BIG_LEAF_SHARE = 0.40f
        private val EXTRA_SHRINKS = floatArrayOf(0.7f, 0.5f)
    }

    init {
        val app = context.applicationContext

        val modelBytes = app.assets.open(MODEL_FILE).use { it.readBytes() }
        session = env.createSession(modelBytes)
        inputName = session.inputNames.first()

        val json = app.assets.open(CLASS_FILE).bufferedReader().use { it.readText() }
        val array = JSONArray(json)
        classNames = List(array.length()) { array.getString(it) }

        require(classNames.isNotEmpty()) { "class_names.json is empty" }
    }

    /**
     * Normal detection plus, when needed, two shrunk passes (zoom 0.7 and 0.5). Overlapping boxes are merged,
     * keeping the most confident one. A confident answer on a normal-sized leaf costs one pass only.
     */
    fun detectMultiScale(bitmap: Bitmap, conf: Float = 0.25f, allowedPrefix: String? = null): List<Detection> {
        val first = detect(bitmap, conf, allowedPrefix, 1f)
        val top = first.firstOrNull()
        val photoArea = bitmap.width.toFloat() * bitmap.height.toFloat()
        val bigLeaf = top != null && (top.right - top.left) * (top.bottom - top.top) > BIG_LEAF_SHARE * photoArea
        val needMore = top == null || top.confidence < CONFIDENT || bigLeaf
        if (!needMore) return first

        val all = ArrayList<Detection>(first)
        for (shrink in EXTRA_SHRINKS) {
            all.addAll(detect(bitmap, conf, allowedPrefix, shrink))
        }
        return mergeOverlapping(all)
    }

    /**
     * @param conf            keep detections at or above this confidence (0..1)
     * @param allowedPrefix   e.g. "Tomato": ignore classes of other crops (a tomato leaf can never be "Corn leaf blight")
     * @param shrink          1 = normal; 0.5 = the photo is shrunk to half size and centred on the grey canvas
     */
    fun detect(bitmap: Bitmap, conf: Float = 0.25f, allowedPrefix: String? = null, shrink: Float = 1f): List<Detection> {

        // Hardware bitmaps cannot be read pixel by pixel.
        val src = if (bitmap.config == Bitmap.Config.HARDWARE) {
            bitmap.copy(Bitmap.Config.ARGB_8888, false)
        } else {
            bitmap
        }

        val w = src.width
        val h = src.height

        // Letterbox: scale to fit inside 640x640 (times the shrink factor), keep the shape, centre it.
        val r = min(SIZE.toFloat() / w, SIZE.toFloat() / h) * shrink
        val newW = (w * r).roundToInt().coerceAtLeast(1)
        val newH = (h * r).roundToInt().coerceAtLeast(1)
        val padLeft = (SIZE - newW) / 2
        val padTop = (SIZE - newH) / 2

        val resized = Bitmap.createScaledBitmap(src, newW, newH, true)
        val pixels = IntArray(newW * newH)
        resized.getPixels(pixels, 0, newW, 0, 0, newW, newH)
        if (resized !== src) resized.recycle()

        // Grey canvas, then copy the photo into the middle. Layout is [1, 3, 640, 640] (R plane, G plane, B plane).
        val plane = SIZE * SIZE
        val input = FloatArray(3 * plane) { PAD }
        for (y in 0 until newH) {
            val outRow = (padTop + y) * SIZE + padLeft
            val inRow = y * newW
            for (x in 0 until newW) {
                val p = pixels[inRow + x]
                val o = outRow + x
                input[o] = ((p shr 16) and 0xFF) / 255f
                input[plane + o] = ((p shr 8) and 0xFF) / 255f
                input[2 * plane + o] = (p and 0xFF) / 255f
            }
        }

        val rows: Array<FloatArray> = OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(input),
            longArrayOf(1, 3, SIZE.toLong(), SIZE.toLong())
        ).use { tensor ->
            session.run(mapOf(inputName to tensor)).use { result ->
                @Suppress("UNCHECKED_CAST")
                (result.get(0).value as Array<Array<FloatArray>>)[0]
            }
        }

        val found = ArrayList<Detection>()
        for (row in rows) {
            val confidence = row[4]
            if (confidence < conf) continue

            val classId = row[5].toInt()
            if (classId !in classNames.indices) continue
            val label = classNames[classId]
            if (allowedPrefix != null && !label.startsWith(allowedPrefix)) continue

            // Back from 640x640 canvas pixels to pixels of the original photo.
            val left = ((row[0] - padLeft) / r).coerceIn(0f, w.toFloat())
            val top = ((row[1] - padTop) / r).coerceIn(0f, h.toFloat())
            val right = ((row[2] - padLeft) / r).coerceIn(0f, w.toFloat())
            val bottom = ((row[3] - padTop) / r).coerceIn(0f, h.toFloat())
            if (right - left < 2f || bottom - top < 2f) continue

            found.add(Detection(label, confidence, left, top, right, bottom))
        }
        return found.sortedByDescending { it.confidence }
    }

    /** Greedy merge: the most confident box wins, boxes that overlap it by more than half are dropped. */
    private fun mergeOverlapping(all: List<Detection>): List<Detection> {
        val kept = ArrayList<Detection>()
        for (d in all.sortedByDescending { it.confidence }) {
            if (kept.none { iou(it, d) > 0.5f }) kept.add(d)
        }
        return kept
    }

    private fun iou(a: Detection, b: Detection): Float {
        val ix = max(0f, min(a.right, b.right) - max(a.left, b.left))
        val iy = max(0f, min(a.bottom, b.bottom) - max(a.top, b.top))
        val inter = ix * iy
        val union = (a.right - a.left) * (a.bottom - a.top) + (b.right - b.left) * (b.bottom - b.top) - inter
        return if (union > 0f) inter / union else 0f
    }

    fun close() {
        session.close()
    }
}