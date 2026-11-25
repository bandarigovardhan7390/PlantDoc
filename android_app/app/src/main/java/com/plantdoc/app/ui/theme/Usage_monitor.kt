package com.plantdoc.app.ui.theme

import android.app.ActivityManager
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Build
import android.os.Debug
import android.os.PowerManager
import android.os.Process
import android.os.SystemClock
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.delay
import java.io.File
import java.util.Locale
import kotlin.math.abs

/** Formats a number for the Usage screen; "n/a" when it was not measured. */
fun usageNum(d: Double, digits: Int = 1): String =
    if (d.isNaN() || d.isInfinite()) "n/a" else String.format(Locale.US, "%." + digits + "f", d)

/**
 * Numbers the app records about its two models while it runs.
 * MainActivity writes them; the Usage screen reads them.
 */
object UsageStats {
    // YOLO (leaf detector)
    @Volatile var yoloLoadMs = -1L
    @Volatile var yoloLoadMemMb = Double.NaN
    @Volatile var yoloLastMs = -1L
    @Volatile var yoloLastCores = Double.NaN
    @Volatile var yoloRuns = 0
    @Volatile var yoloTotalMs = 0L

    // Llama (explanation), last run
    @Volatile var llamaRuns = 0
    @Volatile var llamaTokens = 0
    @Volatile var llamaFirstTokenMs = -1L
    @Volatile var llamaTotalMs = -1L
    @Volatile var llamaTokensPerSec = Double.NaN
    @Volatile var llamaCores = Double.NaN
    @Volatile var llamaMemDeltaMb = Double.NaN
    @Volatile var llamaMah = Double.NaN

    fun recordYoloLoad(ms: Long, memMb: Double) {
        yoloLoadMs = ms
        yoloLoadMemMb = memMb
    }

    fun recordYolo(ms: Long, cpuMs: Long) {
        yoloLastMs = ms
        yoloLastCores = if (ms > 0) cpuMs.toDouble() / ms else Double.NaN
        yoloRuns++
        yoloTotalMs += ms
    }

    /** [uahUsed] is the drop of the battery charge counter during the run (0 when it could not be read). */
    fun recordLlama(tokens: Int, firstMs: Long, totalMs: Long, cpuMs: Long, memDeltaMb: Double, uahUsed: Long) {
        llamaRuns++
        llamaTokens = tokens
        llamaFirstTokenMs = firstMs
        llamaTotalMs = totalMs
        val genMs = totalMs - maxOf(firstMs, 0L)
        llamaTokensPerSec = if (tokens > 1 && genMs > 0) (tokens - 1) * 1000.0 / genMs else Double.NaN
        llamaCores = if (totalMs > 0) cpuMs.toDouble() / totalMs else Double.NaN
        // the memory jump happens when the model is loaded, on the first run
        if (llamaMemDeltaMb.isNaN() || memDeltaMb > llamaMemDeltaMb) llamaMemDeltaMb = memDeltaMb
        llamaMah = if (uahUsed > 0) uahUsed / 1000.0 else Double.NaN
    }
}

/**
 * Reads memory, processor, battery and heat of this phone and this app. No special permission is needed.
 *
 * Honest limits:
 *  - Android does not tell an ordinary app how much power ONE app uses. The power shown is the whole phone's
 *    draw from the battery (screen, radio and the app together). Measure with the phone unplugged.
 *  - The battery charge counter changes in steps, so mAh for one short run is rough.
 */
class UsageMonitor(context: Context) {

    private val app = context.applicationContext
    private val am = app.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
    private val bm = app.getSystemService(Context.BATTERY_SERVICE) as BatteryManager
    private val pm = app.getSystemService(Context.POWER_SERVICE) as PowerManager

    private var lastCpuMs = Process.getElapsedCpuTime()
    private var lastWallMs = SystemClock.elapsedRealtime()
    private val startChargeUah = chargeUah(app)

    companion object {
        /** Memory of this app in MB, the way Android counts it (PSS). */
        fun pssMb(): Double {
            val mi = Debug.MemoryInfo()
            Debug.getMemoryInfo(mi)
            return mi.totalPss / 1024.0
        }

        /** Processor time used by this app so far, in milliseconds. */
        fun cpuMs(): Long = Process.getElapsedCpuTime()

        /** Battery charge left in micro-amp-hours, or -1 when the phone does not report it. */
        fun chargeUah(context: Context): Long {
            val b = context.applicationContext.getSystemService(Context.BATTERY_SERVICE) as BatteryManager
            val v = b.getLongProperty(BatteryManager.BATTERY_PROPERTY_CHARGE_COUNTER)
            return if (v == Long.MIN_VALUE || v <= 0L) -1L else v
        }
    }

    data class Snapshot(
        val deviceName: String,
        val androidVersion: String,
        val cores: Int,
        val ramTotalMb: Double,
        val ramFreeMb: Double,
        val lowMemory: Boolean,
        val pssMb: Double,
        val javaMb: Double,
        val nativeMb: Double,
        val otherMb: Double,
        val cpuCores: Double,
        val threads: Int,
        val batteryPct: Int,
        val plugged: Boolean,
        val voltageV: Double,
        val currentMa: Double,
        val powerW: Double,
        val tempC: Double,
        val usedMah: Double,
        val thermal: String,
        val yoloFileMb: Double,
        val llamaFileMb: Double
    )

    fun snapshot(): Snapshot {
        val mem = ActivityManager.MemoryInfo()
        am.getMemoryInfo(mem)

        val dm = Debug.MemoryInfo()
        Debug.getMemoryInfo(dm)
        val totalPss = dm.totalPss / 1024.0
        val javaPss = dm.dalvikPss / 1024.0
        val nativePss = dm.nativePss / 1024.0

        // average number of processor cores this app kept busy since the last snapshot
        val cpu = Process.getElapsedCpuTime()
        val wall = SystemClock.elapsedRealtime()
        val dWall = wall - lastWallMs
        val cores = if (dWall > 0) (cpu - lastCpuMs).toDouble() / dWall else 0.0
        lastCpuMs = cpu
        lastWallMs = wall

        val battery = app.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val level = battery?.getIntExtra(BatteryManager.EXTRA_LEVEL, -1) ?: -1
        val scale = battery?.getIntExtra(BatteryManager.EXTRA_SCALE, 100) ?: 100
        val pct = if (level >= 0 && scale > 0) level * 100 / scale else -1
        val plugged = (battery?.getIntExtra(BatteryManager.EXTRA_PLUGGED, 0) ?: 0) != 0
        val mv = battery?.getIntExtra(BatteryManager.EXTRA_VOLTAGE, 0) ?: 0
        val tempC = (battery?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, 0) ?: 0) / 10.0

        // Phones differ: some report micro-amps, some milli-amps. Large numbers are taken as micro-amps.
        val raw = bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW)
        val currentMa = if (raw == Int.MIN_VALUE || raw == 0) Double.NaN else if (abs(raw) >= 10000) raw / 1000.0 else raw.toDouble()
        val volts = mv / 1000.0
        val powerW = if (currentMa.isNaN() || volts <= 0.0) Double.NaN else abs(currentMa) / 1000.0 * volts

        val now = chargeUah(app)
        val usedMah = if (startChargeUah > 0 && now > 0) (startChargeUah - now) / 1000.0 else Double.NaN

        return Snapshot(
            deviceName = Build.MANUFACTURER + " " + Build.MODEL,
            androidVersion = "Android " + Build.VERSION.RELEASE + " (API " + Build.VERSION.SDK_INT + ")",
            cores = Runtime.getRuntime().availableProcessors(),
            ramTotalMb = mem.totalMem / 1048576.0,
            ramFreeMb = mem.availMem / 1048576.0,
            lowMemory = mem.lowMemory,
            pssMb = totalPss,
            javaMb = javaPss,
            nativeMb = nativePss,
            otherMb = maxOf(0.0, totalPss - javaPss - nativePss),
            cpuCores = cores,
            threads = threadCount(),
            batteryPct = pct,
            plugged = plugged,
            voltageV = volts,
            currentMa = currentMa,
            powerW = powerW,
            tempC = tempC,
            usedMah = usedMah,
            thermal = thermalText(),
            yoloFileMb = yoloFileMb(),
            llamaFileMb = llamaFileMb()
        )
    }

    /** All numbers as plain text, to paste into the project report. */
    fun report(s: Snapshot): String {
        val sb = StringBuilder()
        sb.append("PlantDoc usage report\n")
        sb.append("Phone: ").append(s.deviceName).append(", ").append(s.androidVersion).append(", ").append(s.cores).append(" cores\n")
        sb.append("RAM: total ").append(usageNum(s.ramTotalMb, 0)).append(" MB, free ").append(usageNum(s.ramFreeMb, 0)).append(" MB\n")
        sb.append("App memory (PSS): ").append(usageNum(s.pssMb)).append(" MB (java ").append(usageNum(s.javaMb))
            .append(", native ").append(usageNum(s.nativeMb)).append(", other ").append(usageNum(s.otherMb)).append(")\n")
        sb.append("CPU now: ").append(usageNum(s.cpuCores, 2)).append(" cores busy, ").append(s.threads).append(" threads\n")
        sb.append("Battery: ").append(s.batteryPct).append("%, plugged in: ").append(s.plugged)
            .append(", ").append(usageNum(s.voltageV, 2)).append(" V, ").append(usageNum(s.currentMa, 0)).append(" mA, whole-phone power ")
            .append(usageNum(s.powerW, 2)).append(" W, ").append(usageNum(s.tempC)).append(" C, thermal ").append(s.thermal).append("\n")
        sb.append("Battery used since screen opened: ").append(usageNum(s.usedMah, 1)).append(" mAh\n")
        sb.append("YOLO: file ").append(usageNum(s.yoloFileMb)).append(" MB, load ").append(UsageStats.yoloLoadMs).append(" ms, memory +")
            .append(usageNum(UsageStats.yoloLoadMemMb)).append(" MB, last run ").append(UsageStats.yoloLastMs).append(" ms using ")
            .append(usageNum(UsageStats.yoloLastCores, 2)).append(" cores, runs ").append(UsageStats.yoloRuns).append("\n")
        sb.append("Llama: file ").append(usageNum(s.llamaFileMb, 0)).append(" MB, memory +").append(usageNum(UsageStats.llamaMemDeltaMb, 0))
            .append(" MB, first word ").append(UsageStats.llamaFirstTokenMs).append(" ms, total ").append(UsageStats.llamaTotalMs)
            .append(" ms, ").append(UsageStats.llamaTokens).append(" pieces, ").append(usageNum(UsageStats.llamaTokensPerSec, 2))
            .append(" per second, ").append(usageNum(UsageStats.llamaCores, 2)).append(" cores, ").append(usageNum(UsageStats.llamaMah, 1)).append(" mAh\n")
        return sb.toString()
    }

    private fun threadCount(): Int {
        return try {
            File("/proc/self/status").readLines().firstOrNull { it.startsWith("Threads:") }
                ?.substringAfter(":")?.trim()?.toInt() ?: -1
        } catch (e: Exception) {
            -1
        }
    }

    private fun thermalText(): String {
        if (Build.VERSION.SDK_INT < 29) return "n/a"
        return when (pm.currentThermalStatus) {
            0 -> "normal"
            1 -> "light"
            2 -> "moderate (may slow down)"
            3 -> "severe (slowing down)"
            4 -> "critical"
            5 -> "emergency"
            6 -> "shutdown"
            else -> "n/a"
        }
    }

    private fun yoloFileMb(): Double {
        return try {
            app.assets.open("leaf_disease_yolo26n.onnx").use { it.available() } / 1048576.0
        } catch (e: Exception) {
            Double.NaN
        }
    }

    private fun llamaFileMb(): Double {
        return try {
            val f = File(app.getExternalFilesDir(null), "llama_3_2_1b_xnnpack_spinquant.pte")
            if (f.exists()) f.length() / 1048576.0 else Double.NaN
        } catch (e: Exception) {
            Double.NaN
        }
    }
}

@Composable
fun UsageScreen(onBack: () -> Unit) {
    val context = LocalContext.current
    val monitor = remember { UsageMonitor(context) }
    var snapshot by remember { mutableStateOf(monitor.snapshot()) }
    val clipboardManager = LocalClipboardManager.current

    LaunchedEffect(Unit) {
        while (true) {
            snapshot = monitor.snapshot()
            delay(1000)
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color(0xFFF6F8F6))
            .verticalScroll(rememberScrollState())
            .padding(16.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                IconButton(onClick = onBack) {
                    Text("←", fontSize = 24.sp, fontWeight = FontWeight.Bold)
                }
                Text(
                    text = "Usage & Performance",
                    fontSize = 20.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color(0xFF1B3B22)
                )
            }
            Button(
                onClick = {
                    val reportText = monitor.report(snapshot)
                    clipboardManager.setText(AnnotatedString(reportText))
                },
                colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF2E7D32)),
                shape = RoundedCornerShape(12.dp)
            ) {
                Text("Copy Report", fontSize = 12.sp)
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        // Device & RAM Card
        UsageMetricCard(title = "📱 Device & Memory (RAM)") {
            MetricRow("Device", snapshot.deviceName)
            MetricRow("OS Version", snapshot.androidVersion)
            MetricRow("CPU Cores (Total)", "${snapshot.cores} cores")
            HorizontalDivider(modifier = Modifier.padding(vertical = 8.dp), color = Color.LightGray.copy(alpha = 0.5f))
            MetricRow("App Memory (PSS)", "${usageNum(snapshot.pssMb)} MB")
            MetricRow("  • Java Heap", "${usageNum(snapshot.javaMb)} MB")
            MetricRow("  • Native Heap", "${usageNum(snapshot.nativeMb)} MB")
            MetricRow("  • Graphics/Other", "${usageNum(snapshot.otherMb)} MB")
            HorizontalDivider(modifier = Modifier.padding(vertical = 8.dp), color = Color.LightGray.copy(alpha = 0.5f))
            MetricRow("System Total RAM", "${usageNum(snapshot.ramTotalMb, 0)} MB")
            MetricRow("System Free RAM", "${usageNum(snapshot.ramFreeMb, 0)} MB")
            if (snapshot.lowMemory) {
                Text("⚠️ Low Memory Warning", color = Color.Red, fontSize = 12.sp, fontWeight = FontWeight.Bold)
            }
        }

        Spacer(modifier = Modifier.height(12.dp))

        // CPU & Battery / Power Card
        UsageMetricCard(title = "⚡ CPU & Power / Battery") {
            MetricRow("Active CPU Cores", "${usageNum(snapshot.cpuCores, 2)} cores busy")
            MetricRow("Active Threads", "${snapshot.threads} threads")
            HorizontalDivider(modifier = Modifier.padding(vertical = 8.dp), color = Color.LightGray.copy(alpha = 0.5f))
            MetricRow("Battery Level", "${snapshot.batteryPct}% (${if (snapshot.plugged) "Plugged in" else "On Battery"})")
            MetricRow("Voltage", "${usageNum(snapshot.voltageV, 2)} V")
            MetricRow("Current Draw", "${usageNum(snapshot.currentMa, 0)} mA")
            MetricRow("Whole-Phone Power", "${usageNum(snapshot.powerW, 2)} W")
            MetricRow("Temperature", "${usageNum(snapshot.tempC)} °C")
            MetricRow("Thermal Status", snapshot.thermal)
            MetricRow("Energy Used (Since open)", "${usageNum(snapshot.usedMah, 1)} mAh")
        }

        Spacer(modifier = Modifier.height(12.dp))

        // YOLO Leaf Detector Model Stats Card
        UsageMetricCard(title = "🍃 YOLO Leaf Detector (ONNX)") {
            MetricRow("Model File Size", "${usageNum(snapshot.yoloFileMb)} MB")
            MetricRow("Model Load Time", if (UsageStats.yoloLoadMs >= 0) "${UsageStats.yoloLoadMs} ms" else "Not loaded")
            MetricRow("Load RAM Jump", "${usageNum(UsageStats.yoloLoadMemMb)} MB")
            MetricRow("Last Inference Time", if (UsageStats.yoloLastMs >= 0) "${UsageStats.yoloLastMs} ms" else "No runs yet")
            MetricRow("Last Run CPU Cores", usageNum(UsageStats.yoloLastCores, 2))
            MetricRow("Total Inferences", "${UsageStats.yoloRuns} runs")
        }

        Spacer(modifier = Modifier.height(12.dp))

        // Llama AI Advisor Model Stats Card
        UsageMetricCard(title = "🦙 Llama 3.2 1B Advisor (ExecuTorch)") {
            MetricRow("Model File Size", "${usageNum(snapshot.llamaFileMb, 0)} MB")
            MetricRow("Model RAM Delta", "${usageNum(UsageStats.llamaMemDeltaMb, 0)} MB")
            MetricRow("Time to 1st Token", if (UsageStats.llamaFirstTokenMs >= 0) "${UsageStats.llamaFirstTokenMs} ms" else "No runs yet")
            MetricRow("Total Generation Time", if (UsageStats.llamaTotalMs >= 0) "${UsageStats.llamaTotalMs} ms" else "No runs yet")
            MetricRow("Tokens Generated", "${UsageStats.llamaTokens} tokens")
            MetricRow("Speed", "${usageNum(UsageStats.llamaTokensPerSec, 2)} tokens/sec")
            MetricRow("CPU Cores Used", usageNum(UsageStats.llamaCores, 2))
            MetricRow("Battery Power Used", "${usageNum(UsageStats.llamaMah, 1)} mAh")
        }

        Spacer(modifier = Modifier.height(20.dp))
    }
}

@Composable
private fun UsageMetricCard(title: String, content: @Composable ColumnScope.() -> Unit) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = Color.White),
        shape = RoundedCornerShape(16.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(title, fontWeight = FontWeight.Bold, fontSize = 16.sp, color = Color(0xFF1B3B22))
            Spacer(modifier = Modifier.height(10.dp))
            content()
        }
    }
}

@Composable
private fun MetricRow(label: String, value: String) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 3.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(label, fontSize = 13.sp, color = Color.Gray)
        Text(value, fontSize = 13.sp, fontWeight = FontWeight.SemiBold, color = Color(0xFF222222))
    }
}