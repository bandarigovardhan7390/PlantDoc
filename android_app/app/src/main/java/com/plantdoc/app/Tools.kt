package com.plantdoc.app

import android.content.Context
import android.speech.tts.TextToSpeech
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import java.util.Locale
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue

/*
 * TOOLS: the small helpers used by MainActivity, joined in one file.
 *   FieldSeverity      - turns the farmer's answers into Mild / Moderate / Severe
 *   FieldCheckSection  - the questions on screen
 *   SpeechHelper       - reads text aloud with the phone's offline voice
 */

// ===================== FieldSeverity.kt =====================
data class FieldCheck(
    val plantsChecked: Int = 10,
    val plantsAffected: Int = 0,
    val spreading: Boolean? = null,
    val recentWetWeather: Boolean = false
)

object FieldSeverity {
    fun estimate(issue: Issue, check: FieldCheck): String {
        // Higher base severity for certain aggressive disease types if needed
        val ratio = if (check.plantsChecked > 0) check.plantsAffected.toFloat() / check.plantsChecked else 0f
        return when {
            ratio > 0.4f || (ratio > 0.2f && check.spreading == true) || (ratio > 0.2f && check.recentWetWeather) -> "Severe"
            ratio > 0.1f || check.spreading == true || check.recentWetWeather -> "Moderate"
            else -> "Mild"
        }
    }

    fun explain(check: FieldCheck, lang: Lang): String {
        val affected = "${check.plantsAffected} of ${check.plantsChecked} plants affected."
        return when (lang) {
            Lang.EN -> affected + (if (check.spreading == true) " Spreading." else "") + (if (check.recentWetWeather) " Recent wet weather." else "")
            Lang.HI -> "${check.plantsAffected} में से ${check.plantsChecked} पौधे प्रभावित।" + (if (check.spreading == true) " फैल रहा है।" else "") + (if (check.recentWetWeather) " हाल ही में नम मौसम।" else "")
            Lang.TE -> "${check.plantsChecked} మొక్కలలో ${check.plantsAffected} ప్రభావితమయ్యాయి." + (if (check.spreading == true) " వ్యాపిస్తోంది." else "") + (if (check.recentWetWeather) " ఇటీవల వర్షం/తేమ." else "")
        }
    }
}

// ===================== FieldCheckSection.kt =====================
/**
 * Replaces the old 3-button Mild/Moderate/Severe picker. The farmer answers 3 quick questions about
 * the PLOT, not just the one leaf in the photo; FieldSeverity.estimate() turns that into the same
 * "Mild"/"Moderate"/"Severe" key the rest of the app (actions, pesticides, reinspect_days) already uses.
 *
 * Usage inside AdviceCard, in place of the old severity ChoiceRow:
 *
 *   var fieldCheck by remember { mutableStateOf(FieldCheck(plantsChecked = 10, plantsAffected = 1, spreading = null, recentWetWeather = false)) }
 *   val severity = FieldSeverity.estimate(issue, fieldCheck)   // use this instead of severityIndex-derived severity
 *
 *   if (!healthy) {
 *       FieldCheckSection(kb, lang, fieldCheck, onChange = { fieldCheck = it })
 *   }
 */
@Composable
fun FieldCheckSection(
    kb: KnowledgeBase,
    lang: Lang,
    check: FieldCheck,
    onChange: (FieldCheck) -> Unit
) {
    Column(modifier = Modifier.fillMaxWidth()) {

        Spacer(modifier = Modifier.height(12.dp))
        Text(kb.text("field_check_title", lang), fontWeight = FontWeight.Bold, fontSize = 15.sp)
        Spacer(modifier = Modifier.height(6.dp))

        // Plants checked / affected - simple stepper rows (avoids opening a numeric keyboard mid-field)
        NumberStepper(
            label = kb.text("plants_checked", lang),
            value = check.plantsChecked,
            min = 1,
            onChange = { onChange(check.copy(plantsChecked = it, plantsAffected = check.plantsAffected.coerceAtMost(it))) }
        )
        NumberStepper(
            label = kb.text("plants_affected", lang),
            value = check.plantsAffected,
            min = 0,
            max = check.plantsChecked,
            onChange = { onChange(check.copy(plantsAffected = it)) }
        )

        Spacer(modifier = Modifier.height(10.dp))
        Text(kb.text("spreading_q", lang), fontSize = 13.sp)
        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            ThreeWayButton(kb.text("spreading_yes", lang), check.spreading == true) { onChange(check.copy(spreading = true)) }
            ThreeWayButton(kb.text("spreading_no", lang), check.spreading == false) { onChange(check.copy(spreading = false)) }
            ThreeWayButton(kb.text("spreading_first", lang), check.spreading == null) { onChange(check.copy(spreading = null)) }
        }

        Spacer(modifier = Modifier.height(10.dp))
        Text(kb.text("wet_q", lang), fontSize = 13.sp)
        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            ThreeWayButton(kb.text("spreading_yes", lang), check.recentWetWeather) { onChange(check.copy(recentWetWeather = true)) }
            ThreeWayButton(kb.text("spreading_no", lang), !check.recentWetWeather) { onChange(check.copy(recentWetWeather = false)) }
        }
    }
}

@Composable
private fun NumberStepper(label: String, value: Int, min: Int = 0, max: Int = 9999, onChange: (Int) -> Unit) {
    Row(
        modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(label, fontSize = 13.sp, modifier = Modifier.weight(1f))
        OutlinedButton(onClick = { if (value > min) onChange(value - 1) }, modifier = Modifier.size(40.dp), contentPadding = PaddingValues(0.dp)) { Text("-") }
        Text(value.toString(), fontSize = 15.sp, fontWeight = FontWeight.Bold, modifier = Modifier.padding(horizontal = 14.dp))
        OutlinedButton(onClick = { if (value < max) onChange(value + 1) }, modifier = Modifier.size(40.dp), contentPadding = PaddingValues(0.dp)) { Text("+") }
    }
}

@Composable
private fun ThreeWayButton(label: String, selected: Boolean, onClick: () -> Unit) {
    if (selected) {
        Button(onClick = onClick, shape = RoundedCornerShape(10.dp), contentPadding = PaddingValues(horizontal = 10.dp)) {
            Text(label, fontSize = 12.sp, maxLines = 1)
        }
    } else {
        OutlinedButton(onClick = onClick, shape = RoundedCornerShape(10.dp), contentPadding = PaddingValues(horizontal = 10.dp)) {
            Text(label, fontSize = 12.sp, maxLines = 1)
        }
    }
}

// ===================== speechhelper.kt =====================
/**
 * Reads advice text aloud using Android's built-in, fully offline TextToSpeech engine.
 * This is what makes the app "read like Google/Gemini" - no network call, no extra model download,
 * because the voice engine (usually Google Speech Services) already ships on almost every Android phone.
 *
 * Known limitation: English and Hindi voices are on virtually every phone. Telugu is NOT guaranteed -
 * budget phones often lack it until the user downloads the Telugu voice pack themselves
 * (Settings > System > Languages > Text-to-speech output > install Telugu). Always check
 * [isAvailable] before showing a Listen button for a language, and tell the user how to get the
 * voice pack if it's missing - do not silently fall back to English speech for Telugu text, since
 * hearing English pronunciation of Telugu script sentences is worse than no audio at all.
 */
class SpeechHelper(context: Context) {

    private val appCtx = context.applicationContext

    @Volatile
    var isReady = false
        private set

    var onReadyListener: (() -> Unit)? = null

    private val tts: TextToSpeech = TextToSpeech(appCtx) { status ->
        if (status == TextToSpeech.SUCCESS) {
            isReady = true
            onReadyListener?.invoke()
        }
    }

    /** Returns the best supported [Locale] for [lang], or null if none available on this device. */
    fun getSupportedLocale(lang: Lang): Locale? {
        if (!isReady) return null
        val candidates = listOf(
            Locale(lang.code, "IN"),
            Locale(lang.code),
            Locale(lang.code, "")
        )
        for (loc in candidates) {
            val res = tts.isLanguageAvailable(loc)
            if (res >= TextToSpeech.LANG_AVAILABLE) {
                return loc
            }
        }
        return null
    }

    /** Call this before showing a "Listen" button for [lang]. */
    fun isAvailable(lang: Lang): Boolean {
        return getSupportedLocale(lang) != null
    }

    /** Speak [text] in [lang]. Stops and replaces anything currently playing. */
    fun speak(text: String, lang: Lang) {
        if (!isReady || text.isBlank()) return
        val loc = getSupportedLocale(lang) ?: return
        val setRes = tts.setLanguage(loc)
        if (setRes == TextToSpeech.LANG_MISSING_DATA || setRes == TextToSpeech.LANG_NOT_SUPPORTED) {
            return
        }
        tts.stop()
        tts.speak(text, TextToSpeech.QUEUE_FLUSH, null, "plantdoc_advice")
    }

    fun stop() {
        if (isReady) tts.stop()
    }

    fun close() {
        if (isReady) tts.stop()
        tts.shutdown()
    }
}

/*
HOW TO WIRE THIS INTO YOUR ADVICE CARD (adapt to your current composable names):

  val speech = remember { SpeechHelper(context.applicationContext) }
  DisposableEffect(speech) { onDispose { speech.close() } }

  var canListen by remember(lang) { mutableStateOf(false) }
  LaunchedEffect(lang) {
      kotlinx.coroutines.delay(300)          // give TTS init a moment on first run
      canListen = speech.isAvailable(lang)
  }

  if (canListen) {
      Button(onClick = {
          val textToRead = if (aiText.isNotEmpty()) aiText
                           else (quick.summary + ". " + quick.steps.joinToString(". "))
          speech.speak(textToRead, lang)
      }) { Text(kb.text("listen", lang)) }     // add a "listen" key to ui{} if not already there
  } else {
      Text(kb.text("listen_unavailable", lang), color = Color.Gray, fontSize = 12.sp)
  }

Reading the Llama paragraph as it streams in works too - just call speech.speak() once generation
finishes (aiSeconds becomes non-null), not on every token, or it will restart mid-sentence repeatedly.
*/


