package com.plantdoc.app

import android.content.Context
import android.os.Handler
import android.os.Looper
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import java.util.Locale

/**
 * Reads one section of the advice aloud, with Pause, Resume and Stop.
 * The phone's own offline voice is used. Android's voice cannot pause in the middle of a sentence,
 * so the text is read sentence by sentence; Pause stops after the current sentence is dropped and
 * Resume starts again from that sentence.
 *
 * Only one section can be read at a time: starting another section stops the first.
 */
class SectionReader(context: Context) {

    private val main = Handler(Looper.getMainLooper())
    private var tts: TextToSpeech? = null
    private var ready = false
    private var queue: List<String> = emptyList()
    private var index = 0
    private var token = 0

    /** Which section is being read or is paused (null = none). The screen reads this to draw its buttons. */
    var activeId by mutableStateOf<String?>(null)
        private set

    /** True while speaking, false while paused. */
    var playing by mutableStateOf(false)
        private set

    init {
        tts = TextToSpeech(context.applicationContext) { status ->
            if (status == TextToSpeech.SUCCESS) {
                ready = true
                tts?.setSpeechRate(0.9f)
                tts?.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                    override fun onStart(utteranceId: String?) {}
                    override fun onDone(utteranceId: String?) { main.post { advance(utteranceId) } }
                    override fun onError(utteranceId: String?) { main.post { advance(utteranceId) } }
                })
            }
        }
    }

    private fun getSupportedLocale(lang: Lang): Locale? {
        val engine = tts ?: return null
        if (!ready) return null
        val candidates = listOf(
            Locale(lang.code, "IN"),
            Locale(lang.code),
            Locale(lang.code, "")
        )
        for (loc in candidates) {
            if (engine.isLanguageAvailable(loc) >= TextToSpeech.LANG_AVAILABLE) {
                return loc
            }
        }
        return null
    }

    /** True when the phone has a voice for this language. */
    fun isAvailable(lang: Lang): Boolean {
        return getSupportedLocale(lang) != null
    }

    /** One button for everything: start, pause, or resume this section. */
    fun toggle(id: String, text: String, lang: Lang) {
        when {
            activeId == id && playing -> pause()
            activeId == id && index < queue.size -> resume()
            else -> start(id, text, lang)
        }
    }

    fun pause() {
        if (!playing) return
        playing = false
        token++            // a late "done" signal from the dropped sentence is ignored
        tts?.stop()
    }

    fun resume() {
        if (activeId == null || playing) return
        playing = true
        token++
        speakCurrent()
    }

    fun stop() {
        token++
        tts?.stop()
        playing = false
        activeId = null
        index = 0
        queue = emptyList()
    }

    fun close() {
        stop()
        tts?.shutdown()
        tts = null
    }

    private fun start(id: String, text: String, lang: Lang) {
        val engine = tts ?: return
        if (!ready) return
        val loc = getSupportedLocale(lang) ?: return
        token++
        engine.stop()
        val sentences = split(text)
        if (sentences.isEmpty()) return
        engine.language = loc
        queue = sentences
        index = 0
        activeId = id
        playing = true
        speakCurrent()
    }

    private fun speakCurrent() {
        val engine = tts ?: return
        if (index >= queue.size) { finish(); return }
        engine.speak(queue[index], TextToSpeech.QUEUE_FLUSH, null, "$token-$index")
    }

    private fun advance(utteranceId: String?) {
        if (!playing || utteranceId != "$token-$index") return
        index++
        if (index >= queue.size) finish() else speakCurrent()
    }

    private fun finish() {
        playing = false
        activeId = null
        index = 0
        queue = emptyList()
    }

    private fun split(text: String): List<String> {
        return text
            .replace("•", ". ")
            .replace("\n", ". ")
            .split(Regex("(?<=[.!?।])\\s+"))
            .map { it.trim() }
            .filter { it.length > 1 }
    }
}