package com.plantdoc.app

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject

data class Pest(val name: String, val why: String)

data class Chemical(val name: String, val useFor: String, val dose: String)

data class Issue(
    val key: String,
    val crop: String,
    val display: String,
    val type: String,
    val summary: String,
    val pests: List<Pest>,
    val effects: List<String>,
    val actions: Map<String, List<String>>,
    val areaSmall: String,
    val areaField: String,
    val escalateIf: String,
    val prevention: List<String>,
    val chemicals: List<Chemical>,
    val sources: List<String>
)

/**
 * Reads knowledge_base.json (English advice + interface texts in en/hi/te) and the optional
 * translations_te.json / translations_hi.json (English sentence -> translated sentence) from the app's assets.
 * Keys of [issues] are exactly the detector's class names.
 */
class KnowledgeBase(context: Context) {

    val common: Map<String, List<String>>
    val issues: Map<String, Issue>
    private val ui: Map<String, Map<String, String>>
    private val translations: Map<String, Map<String, String>>

    companion object {
        const val FILE = "knowledge_base.json"
    }

    init {
        val app = context.applicationContext
        val text = app.assets.open(FILE).bufferedReader().use { it.readText() }
        val root = JSONObject(text)

        common = readListMap(root.getJSONObject("common"))

        val uiMap = LinkedHashMap<String, Map<String, String>>()
        val uiJson = root.getJSONObject("ui")
        val uiKeys = uiJson.keys()
        while (uiKeys.hasNext()) {
            val k = uiKeys.next()
            val o = uiJson.getJSONObject(k)
            val langs = LinkedHashMap<String, String>()
            val lk = o.keys()
            while (lk.hasNext()) {
                val code = lk.next()
                langs[code] = o.getString(code)
            }
            uiMap[k] = langs
        }
        ui = uiMap

        val issueMap = LinkedHashMap<String, Issue>()
        val issuesJson = root.getJSONObject("issues")
        val keys = issuesJson.keys()
        while (keys.hasNext()) {
            val key = keys.next()
            val o = issuesJson.getJSONObject(key)

            val pestArray = o.getJSONArray("pests_to_check")
            val pests = List(pestArray.length()) {
                val p = pestArray.getJSONObject(it)
                Pest(p.getString("name"), p.getString("why"))
            }
            val chemArray = o.getJSONArray("chemical_options")
            val chemicals = List(chemArray.length()) {
                val c = chemArray.getJSONObject(it)
                Chemical(c.getString("name"), c.getString("use_for"), c.getString("dose"))
            }
            val area = o.getJSONObject("area_guidance")

            issueMap[key] = Issue(
                key = key,
                crop = o.getString("crop"),
                display = o.getString("display"),
                type = o.getString("type"),
                summary = o.getString("summary"),
                pests = pests,
                effects = readStrings(o.getJSONArray("effects_if_untreated")),
                actions = readListMap(o.getJSONObject("actions")),
                areaSmall = area.getString("small_plot"),
                areaField = area.getString("field"),
                escalateIf = o.getString("escalate_if"),
                prevention = readStrings(o.getJSONArray("prevention")),
                chemicals = chemicals,
                sources = readStrings(o.getJSONArray("sources"))
            )
        }
        issues = issueMap

        val trMap = LinkedHashMap<String, Map<String, String>>()
        for (lang in Lang.values()) {
            if (lang == Lang.EN) continue
            trMap[lang.code] = loadTranslations(app, "translations_${lang.code}.json")
        }
        translations = trMap
    }

    fun issue(key: String): Issue? = issues[key]

    /** Interface text in the chosen language; falls back to English, then to the key itself. */
    fun text(key: String, lang: Lang): String {
        val m = ui[key] ?: return key
        return m[lang.code] ?: m["en"] ?: key
    }

    /** Translate one English advice sentence. Falls back to the English sentence when no translation exists. */
    fun tr(english: String, lang: Lang): String {
        if (lang == Lang.EN) return english
        return translations[lang.code]?.get(english) ?: english
    }

    private fun loadTranslations(context: Context, file: String): Map<String, String> {
        return try {
            val json = JSONObject(context.assets.open(file).bufferedReader().use { it.readText() })
            val out = LinkedHashMap<String, String>()
            val keys = json.keys()
            while (keys.hasNext()) {
                val k = keys.next()
                out[k] = json.getString(k)
            }
            out
        } catch (e: Exception) {
            emptyMap()
        }
    }

    private fun readStrings(array: JSONArray): List<String> {
        return List(array.length()) { array.getString(it) }
    }

    private fun readListMap(obj: JSONObject): Map<String, List<String>> {
        val out = LinkedHashMap<String, List<String>>()
        val keys = obj.keys()
        while (keys.hasNext()) {
            val k = keys.next()
            out[k] = readStrings(obj.getJSONArray(k))
        }
        return out
    }
}