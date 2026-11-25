package com.plantdoc.app

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.net.Uri
import android.os.Bundle
import android.os.SystemClock
import com.plantdoc.app.ui.theme.UsageMonitor
import com.plantdoc.app.ui.theme.UsageStats
import com.plantdoc.app.ui.theme.UsageScreen
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.setContent
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Canvas as ComposeCanvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.exifinterface.media.ExifInterface
import kotlinx.coroutines.Dispatchers

import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/*
 * PLANTDOC - HOW ONE PHOTO TRAVELS THROUGH THE APP
 *
 *  SCREEN 1  Language        first launch only; the choice is remembered on the phone
 *  SCREEN 2  Home            scan button, last result, how it works
 *  SCREEN 3  Scan            take or choose a photo, press Analyze        (YoloDetector.kt finds the leaf and disease)
 *  SCREEN 4  Result          detection, 4 farmer questions -> severity    (Tools.kt: FieldCheckSection, FieldSeverity)
 *                            advice from the knowledge base               (KnowledgeBase.kt, AdvisoryEngine.kt)
 *                            explanation in words                         (AdvisoryEngine.kt: Llama or reviewed text)
 *                            read aloud                                   (Tools.kt: SpeechHelper)
 *
 * Everything runs on the phone. No internet is used.
 */

private val Green = Color(0xFF246B45)
private val LightGreen = Color(0xFFEAF4EC)
private val Dark = Color(0xFF183B2A)
private val Orange = Color(0xFFB36B00)
private val Red = Color(0xFFC62828)

private const val CONF_THRESHOLD = 0.25f
private const val REVIEW_BELOW = 0.50f
private const val CROP_PREFIX = "Tomato"
private const val MAX_PHOTO_SIDE = 2400

/**
 * Languages where the Llama model writes the explanation. Every other language gets the reviewed,
 * translated text, which is instant and never repeats itself. Llama 3.2 does not officially support
 * Telugu. To try Llama for Hindi again, change this to setOf(Lang.EN, Lang.HI).
 */
private val LLAMA_LANGS = setOf(Lang.EN)

private enum class Screen { LANGUAGE, HOME, SCAN, RESULT, USAGE }

class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            PlantdocApp(onGalleryImage = { uri -> decodeGalleryImage(uri) })
        }
    }

    private fun decodeGalleryImage(uri: Uri): Bitmap? {
        return try {
            val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, bounds) }

            var sample = 1
            val longest = maxOf(bounds.outWidth, bounds.outHeight)
            while (longest / sample > MAX_PHOTO_SIDE) sample *= 2

            val options = BitmapFactory.Options().apply { inSampleSize = sample }
            val decoded = contentResolver.openInputStream(uri)?.use {
                BitmapFactory.decodeStream(it, null, options)
            } ?: return null

            val orientation = contentResolver.openInputStream(uri)?.use {
                ExifInterface(it).getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)
            } ?: ExifInterface.ORIENTATION_NORMAL

            val matrix = Matrix()
            when (orientation) {
                ExifInterface.ORIENTATION_ROTATE_90 -> matrix.postRotate(90f)
                ExifInterface.ORIENTATION_ROTATE_180 -> matrix.postRotate(180f)
                ExifInterface.ORIENTATION_ROTATE_270 -> matrix.postRotate(270f)
                ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> matrix.postScale(-1f, 1f)
                ExifInterface.ORIENTATION_FLIP_VERTICAL -> matrix.postScale(1f, -1f)
            }

            if (matrix.isIdentity) decoded
            else Bitmap.createBitmap(decoded, 0, 0, decoded.width, decoded.height, matrix, true)
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }
    }
}

/** Draws the detection boxes on a copy of the photo. [nameOf] gives the label in the chosen language. */
private fun drawDetections(source: Bitmap, found: List<Detection>, nameOf: (Detection) -> String): Bitmap {
    val out = source.copy(Bitmap.Config.ARGB_8888, true)
    val canvas = Canvas(out)
    val stroke = (maxOf(out.width, out.height) / 250f).coerceAtLeast(3f)
    val boxPaint = Paint().apply { style = Paint.Style.STROKE; strokeWidth = stroke; color = android.graphics.Color.rgb(36, 107, 69); isAntiAlias = true }
    val fillPaint = Paint().apply { style = Paint.Style.FILL; color = android.graphics.Color.rgb(36, 107, 69) }
    val textPaint = Paint().apply { color = android.graphics.Color.WHITE; textSize = stroke * 5f; isAntiAlias = true; isFakeBoldText = true }

    for (d in found) {
        canvas.drawRect(d.left, d.top, d.right, d.bottom, boxPaint)
        val label = "${nameOf(d)} ${(d.confidence * 100).toInt()}%"
        val pad = stroke
        val textWidth = textPaint.measureText(label)
        val labelHeight = textPaint.textSize + 2 * pad
        val labelBottom = if (d.top - labelHeight >= 0f) d.top else d.top + labelHeight
        val labelTop = labelBottom - labelHeight
        canvas.drawRect(d.left, labelTop, d.left + textWidth + 2 * pad, labelBottom, fillPaint)
        canvas.drawText(label, d.left + pad, labelBottom - pad * 1.2f, textPaint)
    }
    return out
}

// =============================================================================================
// APP: state and navigation
// =============================================================================================

@Composable
fun PlantdocApp(onGalleryImage: (Uri) -> Bitmap?) {

    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    // The chosen language is remembered on the phone (nothing is sent anywhere).
    val prefs = remember { context.getSharedPreferences("plantdoc", Context.MODE_PRIVATE) }
    val savedLang = remember { prefs.getInt("lang", -1) }
    var lang by remember { mutableStateOf(if (savedLang in Lang.values().indices) Lang.values()[savedLang] else Lang.EN) }
    var screen by remember { mutableStateOf(if (savedLang in Lang.values().indices) Screen.HOME else Screen.LANGUAGE) }

    var selectedImage by remember { mutableStateOf<Bitmap?>(null) }
    var detector by remember { mutableStateOf<YoloDetector?>(null) }
    var modelStatusKey by remember { mutableStateOf("model_loading") }
    var isModelReady by remember { mutableStateOf(false) }
    var isAnalyzing by remember { mutableStateOf(false) }
    var imageMsgKey by remember { mutableStateOf("msg_none") }
    var detections by remember { mutableStateOf<List<Detection>>(emptyList()) }
    var hasResult by remember { mutableStateOf(false) }
    var inferenceTime by remember { mutableStateOf<Long?>(null) }
    var errorMessage by remember { mutableStateOf<String?>(null) }
    var analysisCount by remember { mutableStateOf(0) }

    val kb = remember { runCatching { KnowledgeBase(context.applicationContext) }.getOrNull() }
    val advisor = remember { LlamaAdvisor(context.applicationContext) }
    val speech = remember { SpeechHelper(context.applicationContext) }

    DisposableEffect(advisor) { onDispose { advisor.close() } }
    DisposableEffect(speech) { onDispose { speech.close() } }

    /** Interface text in the chosen language (KnowledgeBase falls back to English). */
    val tx: (String) -> String = { key -> kb?.text(key, lang) ?: key }

    /** Disease name in the chosen language. */
    val diseaseName: (String) -> String = { label ->
        val key = "issue_$label"
        val v = kb?.text(key, lang)
        if (v == null || v == key) label else v
    }

    val chooseLang: (Lang) -> Unit = { l ->
        lang = l
        prefs.edit().putInt("lang", l.ordinal).apply()
    }

    LaunchedEffect(Unit) {
        try {
            val startMem = UsageMonitor.pssMb()
            val startWall = SystemClock.elapsedRealtime()

            val loaded = withContext(Dispatchers.IO) { YoloDetector(context.applicationContext) }

            val ms = SystemClock.elapsedRealtime() - startWall
            val endMem = UsageMonitor.pssMb()
            UsageStats.recordYoloLoad(ms, maxOf(0.0, endMem - startMem))

            detector = loaded
            isModelReady = true
            modelStatusKey = "model_ready"
        } catch (e: Exception) {
            modelStatusKey = "model_failed"
            errorMessage = e.message ?: "Unknown model loading error"
        }
    }

    DisposableEffect(detector) {
        val active = detector
        onDispose { active?.close() }
    }

    fun resetResult() {
        detections = emptyList()
        hasResult = false
        inferenceTime = null
        errorMessage = null
    }

    // The photo with boxes is redrawn when the language changes, so the box label matches the language.
    val annotatedImage: Bitmap? = remember(selectedImage, detections, hasResult, lang) {
        val img = selectedImage
        if (img != null && hasResult && detections.isNotEmpty()) drawDetections(img, detections) { diseaseName(it.label) }
        else null
    }

    val cameraLauncher = rememberLauncherForActivityResult(ActivityResultContracts.TakePicturePreview()) { bitmap ->
        if (bitmap != null) { selectedImage = bitmap; imageMsgKey = "msg_camera_ok"; resetResult() }
        else imageMsgKey = "msg_camera_cancel"
    }

    val galleryLauncher = rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri != null) {
            scope.launch {
                try {
                    val bitmap = withContext(Dispatchers.IO) { onGalleryImage(uri) }
                    if (bitmap != null) { selectedImage = bitmap; imageMsgKey = "msg_gallery_ok"; resetResult() }
                    else imageMsgKey = "msg_load_fail"
                } catch (e: Exception) {
                    imageMsgKey = "msg_img_fail"
                    errorMessage = e.message
                }
            }
        }
    }

    fun analyzeImage() {
        val image = selectedImage ?: return
        val activeDetector = detector ?: return
        scope.launch {
            isAnalyzing = true
            resetResult()
            try {
                val startCpu = UsageMonitor.cpuMs()
                val startWall = SystemClock.elapsedRealtime()
                val result = withContext(Dispatchers.Default) {
                    val startTime = System.nanoTime()
                    val found = activeDetector.detect(image, conf = CONF_THRESHOLD, allowedPrefix = CROP_PREFIX)
                    val elapsed = (System.nanoTime() - startTime) / 1_000_000
                    Pair(found, elapsed)
                }
                val runMs = SystemClock.elapsedRealtime() - startWall
                val cpuMs = UsageMonitor.cpuMs() - startCpu
                UsageStats.recordYolo(runMs, cpuMs)

                detections = result.first
                inferenceTime = result.second
                hasResult = true
                analysisCount += 1
                imageMsgKey = "msg_done"
                if (result.first.isNotEmpty()) screen = Screen.RESULT
            } catch (e: Exception) {
                errorMessage = e.message ?: "Inference failed. Please try again."
            } finally {
                isAnalyzing = false
            }
        }
    }

    BackHandler(enabled = screen == Screen.SCAN || screen == Screen.RESULT || screen == Screen.USAGE) {
        screen = if (screen == Screen.RESULT) Screen.SCAN else Screen.HOME
    }

    MaterialTheme {
        // Background: drawn by the app itself (no image file, nothing downloaded, works offline)
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(Brush.verticalGradient(listOf(Color(0xFFE3F2E7), Color(0xFFF7FAF7), Color(0xFFD9EEDF))))
        ) {
            LeafDecoration()
            Box(modifier = Modifier.fillMaxSize().statusBarsPadding().navigationBarsPadding()) {
                val top = detections.firstOrNull()
                when (screen) {
                    Screen.LANGUAGE -> LanguageScreen(onPick = { l -> chooseLang(l); screen = Screen.HOME })

                    Screen.HOME -> HomeScreen(
                        tx = tx, lang = lang, onLang = chooseLang,
                        modelReady = isModelReady, modelStatusKey = modelStatusKey,
                        lastName = if (hasResult && top != null) diseaseName(top.label) else null,
                        lastPercent = ((top?.confidence ?: 0f) * 100).toInt(),
                        onScan = { screen = Screen.SCAN },
                        onOpenLast = { screen = Screen.RESULT },
                        onOpenUsage = { screen = Screen.USAGE }
                    )

                    Screen.SCAN -> ScanScreen(
                        tx = tx,
                        image = selectedImage,
                        modelReady = isModelReady, analyzing = isAnalyzing,
                        msgKey = imageMsgKey, error = errorMessage,
                        noLeafFound = hasResult && top == null,
                        onBack = { screen = Screen.HOME },
                        onCamera = { cameraLauncher.launch(null) },
                        onGallery = { galleryLauncher.launch("image/*") },
                        onAnalyze = { analyzeImage() }
                    )

                    Screen.RESULT -> ResultScreen(
                        tx = tx, diseaseName = diseaseName, kb = kb, lang = lang,
                        image = annotatedImage ?: selectedImage,
                        detections = detections, inferenceTime = inferenceTime,
                        analysisCount = analysisCount,
                        advisor = advisor, speech = speech,
                        onBack = { screen = Screen.SCAN },
                        onScanAnother = { screen = Screen.SCAN }
                    )

                    Screen.USAGE -> UsageScreen(onBack = { screen = Screen.HOME })
                }
            }
        }
    }
}

// =============================================================================================
// SCREEN 1: language
// =============================================================================================

@Composable
private fun LanguageScreen(onPick: (Lang) -> Unit) {
    Column(
        modifier = Modifier.fillMaxSize().padding(28.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text("🌿", fontSize = 60.sp)
        Text("PlantDoc", fontSize = 38.sp, fontWeight = FontWeight.Bold, color = Green)
        Spacer(modifier = Modifier.height(8.dp))
        Text(
            "Choose your language  ·  भाषा चुनें  ·  భాష ఎంచుకోండి",
            fontSize = 15.sp, color = Dark, textAlign = TextAlign.Center
        )
        Spacer(modifier = Modifier.height(32.dp))
        Lang.values().forEach { l ->
            Button(
                onClick = { onPick(l) },
                modifier = Modifier.fillMaxWidth().height(64.dp),
                colors = ButtonDefaults.buttonColors(containerColor = Green),
                shape = RoundedCornerShape(18.dp)
            ) { Text(l.label, fontSize = 20.sp, fontWeight = FontWeight.Bold) }
            Spacer(modifier = Modifier.height(14.dp))
        }
    }
}

// =============================================================================================
// SCREEN 2: home
// =============================================================================================

@Composable
private fun HomeScreen(
    tx: (String) -> String,
    lang: Lang,
    onLang: (Lang) -> Unit,
    modelReady: Boolean,
    modelStatusKey: String,
    lastName: String?,
    lastPercent: Int,
    onScan: () -> Unit,
    onOpenLast: () -> Unit,
    onOpenUsage: () -> Unit
) {
    Column(modifier = Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(22.dp)) {

        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text("🌿", fontSize = 36.sp)
                Spacer(modifier = Modifier.width(10.dp))
                Column {
                    Text("PlantDoc", fontSize = 28.sp, fontWeight = FontWeight.Bold, color = Green)
                    Text(tx("tagline"), fontSize = 13.sp, color = Color.Gray)
                }
            }
            IconButton(onClick = onOpenUsage) {
                Text("⚡", fontSize = 24.sp)
            }
        }

        Spacer(modifier = Modifier.height(16.dp))
        ChoiceRow(options = Lang.values().map { it.label }, selected = lang.ordinal, onSelect = { onLang(Lang.values()[it]) })

        Spacer(modifier = Modifier.height(20.dp))
        Text(tx("welcome"), fontSize = 25.sp, fontWeight = FontWeight.Bold, color = Dark)
        Text(tx("subtitle"), color = Color.Gray, fontSize = 15.sp)
        Spacer(modifier = Modifier.height(18.dp))

        Button(
            onClick = onScan,
            modifier = Modifier.fillMaxWidth().height(120.dp),
            colors = ButtonDefaults.buttonColors(containerColor = Green),
            shape = RoundedCornerShape(24.dp)
        ) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Text("📷", fontSize = 36.sp)
                Text(tx("scan_crop"), fontSize = 21.sp, fontWeight = FontWeight.Bold)
                Text(tx("scan_crop_sub"), fontSize = 12.sp)
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        Card(colors = CardDefaults.cardColors(containerColor = LightGreen), shape = RoundedCornerShape(18.dp), modifier = Modifier.fillMaxWidth()) {
            Column(modifier = Modifier.padding(18.dp)) {
                Text(tx("crop_selected"), color = Color.Gray)
                Text("🍅 " + tx("crop_Tomato"), fontSize = 22.sp, fontWeight = FontWeight.Bold, color = Dark)
                Text(tx("leaf_detection"), color = Green)
            }
        }

        if (lastName != null) {
            Spacer(modifier = Modifier.height(12.dp))
            OutlinedButton(onClick = onOpenLast, modifier = Modifier.fillMaxWidth(), shape = RoundedCornerShape(14.dp)) {
                Text(tx("last_result") + ": " + lastName + " (" + lastPercent + "%)", fontSize = 13.sp)
            }
        }

        Spacer(modifier = Modifier.height(12.dp))
        Card(modifier = Modifier.fillMaxWidth(), colors = CardDefaults.cardColors(containerColor = Color.White), shape = RoundedCornerShape(16.dp)) {
            Row(modifier = Modifier.padding(14.dp), verticalAlignment = Alignment.CenterVertically) {
                if (!modelReady && modelStatusKey == "model_loading") {
                    CircularProgressIndicator(modifier = Modifier.size(18.dp), strokeWidth = 2.dp, color = Green)
                    Spacer(modifier = Modifier.width(12.dp))
                }
                Column {
                    Text(tx("model_status"), fontWeight = FontWeight.Bold)
                    Text(tx(modelStatusKey), color = if (modelReady) Green else Color.Gray, fontSize = 13.sp)
                }
            }
        }

        Spacer(modifier = Modifier.height(12.dp))
        Card(modifier = Modifier.fillMaxWidth(), colors = CardDefaults.cardColors(containerColor = Color.White), shape = RoundedCornerShape(16.dp)) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text(tx("how_title"), fontWeight = FontWeight.Bold, fontSize = 16.sp, color = Dark)
                Spacer(modifier = Modifier.height(6.dp))
                Text(tx("how_1"), fontSize = 13.sp)
                Text(tx("how_2"), fontSize = 13.sp)
                Text(tx("how_3"), fontSize = 13.sp)
            }
        }

        Spacer(modifier = Modifier.height(18.dp))
        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.Center, verticalAlignment = Alignment.CenterVertically) {
            Text("● ", color = Green)
            Text(tx("private_note"), color = Color.Gray, fontSize = 12.sp)
        }
        Spacer(modifier = Modifier.height(10.dp))
    }
}

// =============================================================================================
// SCREEN 3: scan
// =============================================================================================

@Composable
private fun ScanScreen(
    tx: (String) -> String,
    image: Bitmap?,
    modelReady: Boolean,
    analyzing: Boolean,
    msgKey: String,
    error: String?,
    noLeafFound: Boolean,
    onBack: () -> Unit,
    onCamera: () -> Unit,
    onGallery: () -> Unit,
    onAnalyze: () -> Unit
) {
    Column(modifier = Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(22.dp)) {

        TextButton(onClick = onBack) { Text("←  " + tx("back"), color = Green, fontSize = 15.sp) }

        Text(tx("analyze_title"), fontSize = 24.sp, fontWeight = FontWeight.Bold, color = Dark)
        Text(tx("photo_tip"), fontSize = 13.sp, color = Color.Gray)
        Spacer(modifier = Modifier.height(16.dp))

        Button(
            onClick = onCamera,
            modifier = Modifier.fillMaxWidth().height(56.dp),
            colors = ButtonDefaults.buttonColors(containerColor = Green),
            shape = RoundedCornerShape(14.dp)
        ) { Text(tx("btn_capture")) }

        Spacer(modifier = Modifier.height(10.dp))

        OutlinedButton(
            onClick = onGallery,
            modifier = Modifier.fillMaxWidth().height(56.dp),
            shape = RoundedCornerShape(14.dp)
        ) { Text(tx("btn_gallery")) }

        Spacer(modifier = Modifier.height(18.dp))

        image?.let { bmp ->
            Card(modifier = Modifier.fillMaxWidth().height(300.dp), shape = RoundedCornerShape(18.dp)) {
                Image(
                    bitmap = bmp.asImageBitmap(),
                    contentDescription = tx("crop_Tomato"),
                    modifier = Modifier.fillMaxSize(),
                    contentScale = ContentScale.Fit
                )
            }
            Spacer(modifier = Modifier.height(12.dp))
            Button(
                onClick = onAnalyze,
                enabled = modelReady && !analyzing,
                modifier = Modifier.fillMaxWidth().height(56.dp),
                colors = ButtonDefaults.buttonColors(containerColor = Green),
                shape = RoundedCornerShape(14.dp)
            ) {
                if (analyzing) {
                    CircularProgressIndicator(modifier = Modifier.size(20.dp), color = Color.White, strokeWidth = 2.dp)
                    Spacer(modifier = Modifier.width(10.dp))
                    Text(tx("analyzing"))
                } else Text(tx("btn_analyze"))
            }
        }

        Spacer(modifier = Modifier.height(14.dp))
        Text(tx(msgKey), color = Color.Gray, fontSize = 13.sp)

        if (noLeafFound) {
            Spacer(modifier = Modifier.height(10.dp))
            Text(tx("det_none"), color = Dark, fontSize = 20.sp, fontWeight = FontWeight.Bold)
            Text(tx("det_none_hint"), color = Orange, fontSize = 13.sp)
        }
        if (error != null) {
            Spacer(modifier = Modifier.height(10.dp))
            Text(tx("error_label") + ": " + error, color = Color.Red, fontSize = 13.sp)
        }
        Spacer(modifier = Modifier.height(20.dp))
    }
}

// =============================================================================================
// SCREEN 4: result (detection, questions, severity, advice, explanation, voice)
// =============================================================================================

@Composable
private fun ResultScreen(
    tx: (String) -> String,
    diseaseName: (String) -> String,
    kb: KnowledgeBase?,
    lang: Lang,
    image: Bitmap?,
    detections: List<Detection>,
    inferenceTime: Long?,
    analysisCount: Int,
    advisor: LlamaAdvisor,
    speech: SpeechHelper,
    onBack: () -> Unit,
    onScanAnother: () -> Unit
) {
    val top = detections.firstOrNull()

    Column(modifier = Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(22.dp)) {

        TextButton(onClick = onBack) { Text("←  " + tx("back"), color = Green, fontSize = 15.sp) }

        image?.let { bmp ->
            Card(modifier = Modifier.fillMaxWidth().height(260.dp), shape = RoundedCornerShape(18.dp)) {
                Image(
                    bitmap = bmp.asImageBitmap(),
                    contentDescription = tx("crop_Tomato"),
                    modifier = Modifier.fillMaxSize(),
                    contentScale = ContentScale.Fit
                )
            }
            Spacer(modifier = Modifier.height(14.dp))
        }

        if (top != null) {
            Card(modifier = Modifier.fillMaxWidth(), colors = CardDefaults.cardColors(containerColor = Color.White), shape = RoundedCornerShape(16.dp)) {
                Column(modifier = Modifier.padding(18.dp)) {
                    Text(tx("det_title"), fontWeight = FontWeight.Bold, fontSize = 16.sp, color = Color.Gray)
                    Spacer(modifier = Modifier.height(6.dp))
                    Text(tx("det_top"), color = Color.Gray, fontSize = 13.sp)
                    Text(diseaseName(top.label), color = Dark, fontSize = 21.sp, fontWeight = FontWeight.Bold)
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(tx("confidence") + ": " + (top.confidence * 100).toInt() + "%", color = Green, fontWeight = FontWeight.Bold)

                    val perClass = detections.groupingBy { diseaseName(it.label) }.eachCount().entries
                        .joinToString(", ") { it.key + " × " + it.value }
                    Spacer(modifier = Modifier.height(5.dp))
                    Text(tx("leaves_found") + ": " + perClass, color = Color.Gray, fontSize = 13.sp)
                    inferenceTime?.let {
                        Spacer(modifier = Modifier.height(5.dp))
                        Text(tx("inference_time") + ": " + it + " ms", color = Color.Gray, fontSize = 13.sp)
                    }
                    Spacer(modifier = Modifier.height(10.dp))
                    if (top.confidence < REVIEW_BELOW) {
                        Text(tx("det_not_sure"), color = Orange, fontSize = 13.sp, fontWeight = FontWeight.Medium)
                    } else {
                        Text(tx("det_ok"), color = Green, fontSize = 13.sp)
                    }
                }
            }

            Spacer(modifier = Modifier.height(16.dp))
            val topIssue = kb?.issue(top.label)
            if (kb != null && topIssue != null) {
                key(analysisCount) {
                    AdviceCard(kb = kb, issue = topIssue, confidence = top.confidence, lang = lang, advisor = advisor, speech = speech)
                }
            } else {
                Text(
                    "Advice not available: knowledge_base.json is missing from the app assets, or has no entry for ${top.label}.",
                    color = Orange, fontSize = 13.sp
                )
            }
        }

        Spacer(modifier = Modifier.height(16.dp))
        OutlinedButton(onClick = onScanAnother, modifier = Modifier.fillMaxWidth().height(52.dp), shape = RoundedCornerShape(14.dp)) {
            Text(tx("scan_another"))
        }
        Spacer(modifier = Modifier.height(14.dp))
        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.Center, verticalAlignment = Alignment.CenterVertically) {
            Text("● ", color = Green)
            Text(tx("footer_offline"), color = Color.Gray, fontSize = 13.sp)
        }
        Spacer(modifier = Modifier.height(10.dp))
    }
}

// ---------------------------------------------------------------------------------------------
// ADVICE CARD: 4 questions -> severity | knowledge-base advice | explanation | read aloud
// ---------------------------------------------------------------------------------------------

@Composable
private fun AdviceCard(
    kb: KnowledgeBase,
    issue: Issue,
    confidence: Float,
    lang: Lang,
    advisor: LlamaAdvisor,
    speech: SpeechHelper
) {
    val scope = rememberCoroutineScope()

    var smallPlot by remember { mutableStateOf(true) }
    // 25 plants = the NIPHM scouting method: 5 spots x 5 plants
    var fieldCheck by remember { mutableStateOf(FieldCheck(plantsChecked = 25, plantsAffected = 0, spreading = null, recentWetWeather = false)) }
    var aiText by remember { mutableStateOf("") }
    var aiBusy by remember { mutableStateOf(false) }
    var aiNote by remember { mutableStateOf("") }
    var aiSeconds by remember { mutableStateOf<Long?>(null) }
    var canListen by remember { mutableStateOf(speech.isAvailable(lang)) }

    val healthy = issue.type == "healthy"
    val severity = if (healthy) "Healthy" else FieldSeverity.estimate(issue, fieldCheck)
    val useReviewedText = lang !in LLAMA_LANGS || lang.llamaName == null

    val input = AdvisoryInput(issue, confidence, severity, smallPlot, lang)
    val quick = Advisory.quick(kb, input)
    val appCtx = LocalContext.current.applicationContext

    DisposableEffect(speech, lang) {
        speech.onReadyListener = {
            canListen = speech.isAvailable(lang)
        }
        canListen = speech.isAvailable(lang)
        onDispose {
            speech.onReadyListener = null
        }
    }

    LaunchedEffect(lang) {
        // a text written in the old language must not stay on screen
        aiText = ""
        aiNote = ""
        aiSeconds = null
        canListen = speech.isAvailable(lang)
    }

    Card(modifier = Modifier.fillMaxWidth(), colors = CardDefaults.cardColors(containerColor = Color.White), shape = RoundedCornerShape(16.dp)) {
        Column(modifier = Modifier.padding(18.dp)) {

            Text(kb.text("advice_title", lang), fontWeight = FontWeight.Bold, fontSize = 18.sp)

            SectionTitle(kb.text("plot_size", lang))
            ChoiceRow(
                options = listOf(kb.text("small_plot", lang), kb.text("field", lang)),
                selected = if (smallPlot) 0 else 1,
                onSelect = { smallPlot = (it == 0) }
            )

            if (!healthy) {
                FieldCheckSection(kb = kb, lang = lang, check = fieldCheck, onChange = { fieldCheck = it })
                Spacer(modifier = Modifier.height(10.dp))
                val sevColor = when (severity) { "Mild" -> Green; "Moderate" -> Orange; else -> Red }
                Text(
                    kb.text("estimated_severity", lang) + ": " + kb.text("severity_$severity", lang),
                    fontWeight = FontWeight.Bold, fontSize = 15.sp, color = sevColor
                )
                Text(FieldSeverity.explain(fieldCheck, lang), fontSize = 12.sp, color = Color.Gray)
            }

            SectionTitle(kb.text("what_found", lang))
            val shownName = kb.text("issue_" + issue.key, lang).let { if (it == "issue_" + issue.key) issue.display else it }
            Text(
                shownName + "  (" + kb.text("confidence", lang) + ": " + (confidence * 100).toInt() + "%)",
                fontWeight = FontWeight.Bold, color = Dark, fontSize = 15.sp
            )
            Spacer(modifier = Modifier.height(4.dp))
            Text(quick.summary, fontSize = 13.sp, color = Color(0xFF333333))

            if (confidence < Advisory.LOW_CONFIDENCE) {
                Spacer(modifier = Modifier.height(6.dp))
                Text(kb.text("not_sure", lang), color = Orange, fontSize = 13.sp, fontWeight = FontWeight.Medium)
            }

            if (healthy) {
                SectionTitle(kb.text("how_serious", lang))
                Text(kb.text("healthy_note", lang), fontSize = 13.sp, color = Green)
            }

            Spacer(modifier = Modifier.height(10.dp))
            if (canListen) {
                OutlinedButton(
                    onClick = { speech.speak((listOf(quick.summary) + quick.steps + listOf(quick.whenToCall)).joinToString(". "), lang) },
                    shape = RoundedCornerShape(10.dp)
                ) { Text(kb.text("listen_advice", lang)) }
            } else {
                Text(kb.text("listen_unavailable", lang), fontSize = 11.sp, color = Color.Gray)
            }

            SectionTitle(kb.text("what_to_do", lang))
            BulletList(quick.steps)

            if (quick.prevention.isNotEmpty()) {
                SectionTitle(kb.text("prevention", lang))
                BulletList(quick.prevention)
            }

            SectionTitle(kb.text("pests_to_check", lang))
            BulletList(quick.pests)

            if (quick.effects.isNotEmpty()) {
                SectionTitle(kb.text("if_untreated", lang))
                BulletList(quick.effects)
            }

            SectionTitle(kb.text("plot_advice", lang))
            Text(quick.plotTip, fontSize = 13.sp, color = Color(0xFF333333))

            SectionTitle(kb.text("chem_options", lang))
            when {
                issue.chemicals.isEmpty() -> Text(kb.text("chem_none", lang), color = Color.Gray, fontSize = 13.sp)
                !quick.showChemicals -> Text(kb.text("chem_wait", lang), color = Color.Gray, fontSize = 13.sp)
                else -> {
                    Text(kb.text("chem_disclaimer", lang), fontSize = 11.sp, color = Color.Gray)
                    Spacer(modifier = Modifier.height(6.dp))
                    quick.chemicals.forEach { chem ->
                        Column(modifier = Modifier.padding(bottom = 8.dp)) {
                            Text(chem.name, fontWeight = FontWeight.Bold, fontSize = 13.sp, color = Dark)
                            Text(kb.text("dose_label", lang) + ": " + chem.dose, fontSize = 12.sp, color = Color(0xFF333333))
                            Text(chem.useFor, fontSize = 12.sp, color = Color.Gray)
                        }
                    }
                }
            }

            SectionTitle(kb.text("when_to_call", lang))
            Text(
                quick.whenToCall, fontSize = 13.sp,
                fontWeight = if (quick.urgent) FontWeight.Bold else FontWeight.Normal,
                color = if (quick.urgent) Red else Color(0xFF333333)
            )

            if (quick.sources.isNotEmpty()) {
                SectionTitle(kb.text("source_title", lang))
                quick.sources.forEach { Text("• $it", fontSize = 11.sp, color = Color.Gray) }
            }

            // ----- explanation in words
            Spacer(modifier = Modifier.height(16.dp))
            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button(
                    onClick = {
                        if (useReviewedText) {
                            // Reviewed, translated text: instant, same wording every time.
                            aiText = Advisory.narrative(kb, input)
                            aiNote = ""
                            aiSeconds = 0L
                        } else if (!advisor.filesFound()) {
                            aiNote = advisor.missingFilesMessage()
                        } else if (!aiBusy) {
                            aiText = ""
                            aiSeconds = null
                            aiNote = kb.text("llama_starting", lang)
                            aiBusy = true
                            val prompt = Advisory.buildPrompt(kb, input)
                            scope.launch {
                                val startCpu = UsageMonitor.cpuMs()
                                val startWall = SystemClock.elapsedRealtime()
                                val startCharge = UsageMonitor.chargeUah(appCtx)
                                val startMem = UsageMonitor.pssMb()
                                var firstTokenMs = -1L
                                var tokens = 0

                                val error = withContext(Dispatchers.Default) {
                                    advisor.generate(prompt) { token ->
                                        if (tokens == 0) {
                                            firstTokenMs = SystemClock.elapsedRealtime() - startWall
                                        }
                                        tokens++
                                        if (aiNote.isNotEmpty()) aiNote = ""
                                        aiText += token
                                    }
                                }

                                val totalMs = SystemClock.elapsedRealtime() - startWall
                                val cpuMs = UsageMonitor.cpuMs() - startCpu
                                val endCharge = UsageMonitor.chargeUah(appCtx)
                                val uahUsed = if (startCharge > 0 && endCharge > 0 && startCharge >= endCharge) startCharge - endCharge else 0L
                                val endMem = UsageMonitor.pssMb()

                                UsageStats.recordLlama(tokens, firstTokenMs, totalMs, cpuMs, maxOf(0.0, endMem - startMem), uahUsed)

                                aiSeconds = totalMs / 1000
                                if (error != null) aiNote = error
                                aiBusy = false
                            }
                        }
                    },
                    enabled = !aiBusy,
                    modifier = Modifier.weight(1f).height(50.dp),
                    colors = ButtonDefaults.buttonColors(containerColor = Green),
                    shape = RoundedCornerShape(12.dp)
                ) { Text(kb.text("explain_ai", lang)) }

                if (aiBusy) {
                    OutlinedButton(onClick = { advisor.stop() }, modifier = Modifier.height(50.dp), shape = RoundedCornerShape(12.dp)) {
                        Text(kb.text("stop", lang))
                    }
                }
            }

            if (aiNote.isNotEmpty()) {
                Spacer(modifier = Modifier.height(8.dp))
                Text(aiNote, color = Orange, fontSize = 12.sp)
            }

            if (aiText.isNotEmpty()) {
                SectionTitle(kb.text("ai_title", lang))
                Text(aiText, fontSize = 14.sp, color = Dark)
                aiSeconds?.let {
                    Spacer(modifier = Modifier.height(4.dp))
                    Text(
                        if (useReviewedText) kb.text("reviewed_note", lang) else kb.text("llama_time", lang).replace("{s}", it.toString()),
                        fontSize = 11.sp, color = Color.Gray
                    )
                }

                Spacer(modifier = Modifier.height(8.dp))
                if (canListen) {
                    OutlinedButton(onClick = { speech.speak(aiText, lang) }, shape = RoundedCornerShape(10.dp)) {
                        Text(kb.text("listen", lang))
                    }
                }
            }

            Spacer(modifier = Modifier.height(14.dp))
            Text(kb.text("disclaimer", lang), fontSize = 11.sp, color = Color.Gray)
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Small shared pieces
// ---------------------------------------------------------------------------------------------

/** Soft leaf-coloured shapes behind every screen. Drawn in code, so no image file is needed. */
@Composable
private fun LeafDecoration() {
    ComposeCanvas(modifier = Modifier.fillMaxSize()) {
        drawCircle(color = Color(0x1A246B45), radius = size.width * 0.55f, center = Offset(size.width * 0.95f, -size.height * 0.02f))
        drawCircle(color = Color(0x14246B45), radius = size.width * 0.45f, center = Offset(-size.width * 0.05f, size.height * 0.98f))
        drawOval(color = Color(0x12246B45), topLeft = Offset(size.width * 0.5f, size.height * 0.62f), size = Size(size.width * 0.8f, size.height * 0.16f))
    }
}

@Composable
private fun SectionTitle(text: String) {
    Spacer(modifier = Modifier.height(12.dp))
    Text(text, fontWeight = FontWeight.Bold, fontSize = 15.sp, color = Dark)
    Spacer(modifier = Modifier.height(4.dp))
}

@Composable
private fun BulletList(items: List<String>) {
    items.forEach {
        Text("• $it", fontSize = 13.sp, color = Color(0xFF333333), modifier = Modifier.padding(bottom = 3.dp))
    }
}

@Composable
private fun ChoiceRow(options: List<String>, selected: Int, onSelect: (Int) -> Unit) {
    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        options.forEachIndexed { index, label ->
            if (index == selected) {
                Button(
                    onClick = { onSelect(index) }, modifier = Modifier.weight(1f),
                    colors = ButtonDefaults.buttonColors(containerColor = Green),
                    shape = RoundedCornerShape(10.dp), contentPadding = PaddingValues(horizontal = 4.dp)
                ) { Text(label, fontSize = 13.sp, maxLines = 1) }
            } else {
                OutlinedButton(
                    onClick = { onSelect(index) }, modifier = Modifier.weight(1f),
                    shape = RoundedCornerShape(10.dp), contentPadding = PaddingValues(horizontal = 4.dp)
                ) { Text(label, fontSize = 13.sp, maxLines = 1) }
            }
        }
    }
}