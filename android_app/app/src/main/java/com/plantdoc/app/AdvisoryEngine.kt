package com.plantdoc.app

import android.content.Context
import org.pytorch.executorch.extension.llm.LlmCallback
import org.pytorch.executorch.extension.llm.LlmGenerationConfig
import org.pytorch.executorch.extension.llm.LlmModule
import java.io.File
import java.util.Locale

/** llamaName = language Llama is asked to write in. null = Llama does not write this language (Telugu for now). */
enum class Lang(val code: String, val label: String, val llamaName: String?) {
    EN("en", "English", "English"),
    HI("hi", "हिन्दी", "Hindi (Devanagari script)"),
    TE("te", "తెలుగు", null)
}

data class AdvisoryInput(
    val issue: Issue,
    val confidence: Float,
    val severity: String,      // "Mild", "Moderate" or "Severe" (ignored for healthy leaves)
    val smallPlot: Boolean,
    val lang: Lang
)

/** Instant advice built only from the knowledge base, already translated. Always shown; Llama adds an explanation on top. */
data class QuickAdvice(
    val summary: String,
    val steps: List<String>,
    val prevention: List<String>,
    val chemicals: List<Chemical>,
    val showChemicals: Boolean,
    val pests: List<String>,
    val effects: List<String>,
    val plotTip: String,
    val whenToCall: String,
    val urgent: Boolean,
    val sources: List<String>
)

object Advisory {

    const val LOW_CONFIDENCE = 0.5f

    private fun severityKey(input: AdvisoryInput): String {
        return if (input.issue.type == "healthy") "Healthy" else input.severity
    }

    private fun rawSteps(kb: KnowledgeBase, input: AdvisoryInput): List<String> {
        val key = severityKey(input)
        return if (key == "Healthy") {
            input.issue.actions["Healthy"].orEmpty()
        } else {
            kb.common[key].orEmpty() + input.issue.actions[key].orEmpty()
        }
    }

    /** Spray options are only shown for a real problem (not Mild), when the app is sure enough. */
    private fun chemicalsVisible(input: AdvisoryInput): Boolean {
        return input.issue.type != "healthy" &&
                input.severity != "Mild" &&
                input.confidence >= LOW_CONFIDENCE &&
                input.issue.chemicals.isNotEmpty()
    }

    fun quick(kb: KnowledgeBase, input: AdvisoryInput): QuickAdvice {
        val issue = input.issue
        val lang = input.lang
        val low = input.confidence < LOW_CONFIDENCE
        val severe = issue.type != "healthy" && input.severity == "Severe"

        val call = when {
            low -> kb.text("not_sure", lang)
            severe -> kb.text("call_expert_severe", lang) + " " + kb.tr(issue.escalateIf, lang)
            else -> kb.tr(issue.escalateIf, lang)
        }

        return QuickAdvice(
            summary = kb.tr(issue.summary, lang),
            steps = rawSteps(kb, input).map { kb.tr(it, lang) },
            prevention = issue.prevention.map { kb.tr(it, lang) },
            chemicals = issue.chemicals.map { Chemical(it.name, kb.tr(it.useFor, lang), kb.tr(it.dose, lang)) },
            showChemicals = chemicalsVisible(input),
            pests = issue.pests.map { kb.tr(it.name, lang) + ": " + kb.tr(it.why, lang) },
            effects = issue.effects.map { kb.tr(it, lang) },
            plotTip = kb.tr(if (input.smallPlot) issue.areaSmall else issue.areaField, lang),
            whenToCall = call,
            urgent = low || severe,
            sources = issue.sources
        )
    }

    /**
     * A short explanation paragraph built from the reviewed, translated knowledge-base text. It works instantly and
     * offline in every language, including Telugu, where Llama 3.2 1B cannot write reliably yet.
     */
    fun narrative(kb: KnowledgeBase, input: AdvisoryInput): String {
        val lang = input.lang
        val issue = input.issue
        val healthy = issue.type == "healthy"
        val low = input.confidence < LOW_CONFIDENCE
        val pct = (input.confidence * 100).toInt().toString()
        val q = quick(kb, input)

        val parts = ArrayList<String>()
        parts.add(
            kb.text("narr_found", lang)
                .replace("{crop}", kb.text("crop_${issue.crop}", lang))
                .replace("{issue}", kb.tr(issue.display, lang))
                .replace("{pct}", pct)
        )
        if (low) parts.add(kb.text("not_sure", lang))
        if (!healthy) {
            parts.add(kb.text("narr_sev", lang).replace("{sev}", kb.text("severity_${input.severity}", lang)))
        }
        val steps = q.steps.take(3).joinToString(" ")
        if (steps.isNotEmpty()) parts.add(kb.text("narr_do", lang).replace("{steps}", steps))
        val pestNames = issue.pests.take(3).joinToString(", ") { kb.tr(it.name, lang) }
        if (pestNames.isNotEmpty()) parts.add(kb.text("narr_pest", lang).replace("{pests}", pestNames))
        parts.add(kb.text("narr_call", lang).replace("{call}", q.whenToCall))
        return parts.joinToString(" ")
    }

    /**
     * Llama 3 chat prompt (English facts). Everything Llama may say is inside FACTS, so the answer is grounded.
     * Doses are NOT given to Llama: the card shows the reference doses from the knowledge base.
     */
    fun buildPrompt(kb: KnowledgeBase, input: AdvisoryInput): String {
        val issue = input.issue
        val healthy = issue.type == "healthy"
        val langName = input.lang.llamaName ?: "English"
        val pct = (input.confidence * 100).toInt()

        val system = "You are PlantDoc, a careful and friendly farm advisor for smallholder farmers. " +
                "Use ONLY the FACTS in the user message. Never invent diseases, pests, products, doses or numbers. " +
                "Never state any dose. If the farmer needs something that is not in the FACTS, tell them to ask the local agriculture extension officer. " +
                "Use simple words and short sentences. Do not use markdown, bullet symbols or special tokens. " +
                "Write in $langName, in 5 to 7 sentences, in this order: " +
                "what was found and how sure the app is; how serious it is; what to do first; " +
                "which pests to check; how to prevent it; when to call the extension officer."

        val facts = StringBuilder()
        facts.append("FACTS\n")
        facts.append("Crop: ").append(issue.crop).append("\n")
        facts.append("Finding: ").append(issue.display).append("\n")
        facts.append("App confidence: ").append(pct).append("%")
        if (input.confidence < LOW_CONFIDENCE) facts.append(" (LOW: say clearly that the app is not sure and ask for a clearer photo)")
        facts.append("\n")
        if (!healthy) {
            facts.append("Severity: ").append(input.severity).append(" (a rough estimate of the leaf area affected)\n")
        }
        facts.append("Plot: ").append(if (input.smallPlot) "small plot" else "field").append("\n")
        facts.append("What it is: ").append(issue.summary).append("\n")
        facts.append("Actions:\n")
        rawSteps(kb, input).take(6).forEach { facts.append("- ").append(it).append("\n") }
        if (issue.prevention.isNotEmpty()) {
            facts.append("Prevention:\n")
            issue.prevention.take(3).forEach { facts.append("- ").append(it).append("\n") }
        }
        facts.append("Pests to check:\n")
        issue.pests.take(3).forEach { facts.append("- ").append(it.name).append(": ").append(it.why).append("\n") }
        if (chemicalsVisible(input)) {
            facts.append("Spray options listed in the app (names only): ")
                .append(issue.chemicals.joinToString("; ") { it.name })
                .append(". Tell the farmer to check the product label and the extension officer before spraying.\n")
        }
        facts.append("Call the extension officer if: ").append(issue.escalateIf).append("\n")
        if (!healthy && input.severity == "Severe") {
            facts.append("This is Severe: tell the farmer to contact the extension officer today.\n")
        }
        facts.append("\nExplain this to the farmer.")

        return "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" + system + "<|eot_id|>" +
                "<|start_header_id|>user<|end_header_id|>\n\n" + facts + "<|eot_id|>" +
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
    }
}

/**
 * Runs Llama 3.2 1B on the phone with ExecuTorch. This is the ADVISORY pipeline.
 * (The leaf detector uses ONNX Runtime and is a separate pipeline.)
 *
 * Put llama_3_2_1b_xnnpack_spinquant.pte and tokenizer.json in the app's files folder:
 *   /sdcard/Android/data/com.plantdoc.app/files/
 */
class LlamaAdvisor(private val context: Context) {

    companion object {
        const val MODEL_FILE = "llama_3_2_1b_xnnpack_spinquant.pte"
        const val TOKENIZER_FILE = "tokenizer.json"
        private const val TEMPERATURE = 0.2f
        private const val SEQ_LEN = 1200
        private val SPECIAL_TOKENS = listOf("<|eot_id|>", "<|end_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|begin_of_text|>")
    }

    private var module: LlmModule? = null
    private var loadSeconds = 0.0
    private var tokenCount = 0
    private var genSeconds = 0.0

    private fun findFile(name: String): File? {
        val candidates = listOf(
            File(context.getExternalFilesDir(null), name),
            File("/data/local/tmp/llama/$name")
        )
        return candidates.firstOrNull { it.exists() && it.length() > 0 }
    }

    fun filesFound(): Boolean = findFile(MODEL_FILE) != null && findFile(TOKENIZER_FILE) != null

    fun missingFilesMessage(): String {
        val dir = context.getExternalFilesDir(null)?.absolutePath ?: "the app files folder"
        return "Llama files not found. Copy $MODEL_FILE and $TOKENIZER_FILE to $dir"
    }

    /** "Loaded in 12.3 s | 180 tokens in 40.2 s (4.5 tokens/s)" for the last run. */
    fun statsText(): String {
        val rate = if (genSeconds > 0) tokenCount / genSeconds else 0.0
        return String.format(
            Locale.US,
            "Model load %.1f s | %d tokens in %.1f s (%.1f tokens/s)",
            loadSeconds, tokenCount, genSeconds, rate
        )
    }

    /** Returns null when loaded, otherwise an error message. */
    @Synchronized
    private fun ensureLoaded(): String? {
        if (module != null) return null
        val model = findFile(MODEL_FILE)
        val tokenizer = findFile(TOKENIZER_FILE)
        if (model == null || tokenizer == null) return missingFilesMessage()

        return try {
            val start = System.nanoTime()
            val m = LlmModule(model.absolutePath, tokenizer.absolutePath, TEMPERATURE)
            m.load()    // in ExecuTorch 1.5 this returns nothing and throws if the model cannot be loaded
            loadSeconds = (System.nanoTime() - start) / 1_000_000_000.0
            module = m
            null
        } catch (e: Throwable) {
            "Llama failed to load: ${e.message}. The .pte file and the ExecuTorch library version may not match."
        }
    }

    /**
     * Blocking: call from a background thread. Streams cleaned tokens to [onToken].
     * Returns null on success, otherwise an error message.
     */
    fun generate(prompt: String, onToken: (String) -> Unit): String? {
        val loadError = ensureLoaded()
        if (loadError != null) return loadError
        val m = module ?: return "Llama is not loaded"

        var error: String? = null
        var tokens = 0
        val callback = object : LlmCallback {
            override fun onResult(token: String) {
                var clean = token
                for (special in SPECIAL_TOKENS) clean = clean.replace(special, "")
                if (clean.isNotEmpty()) {
                    tokens += 1
                    onToken(clean)
                }
            }

            override fun onStats(statsJson: String) {
                // Speed statistics arrive here when generation finishes.
            }

            override fun onError(errorCode: Int, message: String) {
                error = "Llama error $errorCode: $message"
            }
        }

        return try {
            m.resetContext()
            val config = LlmGenerationConfig.create()
                .seqLen(SEQ_LEN)
                .temperature(TEMPERATURE)
                .echo(false)
                .numBos(0)          // the prompt already starts with <|begin_of_text|>
                .build()
            val start = System.nanoTime()
            m.generate(prompt, config, callback)
            genSeconds = (System.nanoTime() - start) / 1_000_000_000.0
            tokenCount = tokens
            error
        } catch (e: Throwable) {
            "Llama failed while generating: ${e.message}"
        }
    }

    fun stop() {
        try {
            module?.stop()
        } catch (e: Throwable) {
            // ignore
        }
    }

    fun close() {
        stop()
        module = null
    }
}