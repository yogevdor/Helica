package ai.helica.asr

/**
 * Swappable speech-recognition engine. Both Plan A (Nemotron via NeMo-Speech.cpp) and
 * Plan B (Whisper via whisper.cpp) implement this, so the UI and the glasses link never
 * depend on which model runs.
 *
 * Threading: feedAudio() is called from the single inference thread. Callbacks may be invoked
 * on that thread; the caller posts them to the main thread.
 */
interface SpeechEngine {
    /** 16 kHz mono PCM, float in [-1, 1]. Frames already passed through VAD (plus ~0.5–1 s after speech). */
    fun feedAudio(samples: FloatArray)

    /** Called whenever the current segment's text changes. See [CaptionUpdate]. */
    var onUpdate: (CaptionUpdate) -> Unit

    /** Load the model and warm up (~4 s for Nemotron on Snapdragon 695). Call once at app start. */
    fun load(modelPath: String, config: EngineConfig = EngineConfig())

    /** Flush pending audio at end of utterance (VAD detected a long pause). Ends the current segment. */
    fun endOfUtterance()

    fun release()
}

/**
 * One caption update for one segment (a line on screen / one BLE diff).
 *
 * Texts are cumulative for the segment, not deltas.
 * - Nemotron (cache-aware RNNT) only appends: [stableText] grows, [unstableText] is at most the last
 *   incomplete word.
 * - Whisper may rewrite [unstableText] until it commits it into [stableText].
 */
data class CaptionUpdate(
    val segmentId: Long,
    /** Never changes once sent (shown normal). */
    val stableText: String,
    /** May change in the next update (shown grey / italic). */
    val unstableText: String,
    /** Word timing and confidence, if the engine provides them (check NeMo-Speech.cpp's C ABI). */
    val words: List<Word> = emptyList(),
    /** True on the last update of this segment. */
    val segmentEnded: Boolean = false,
)

data class Word(
    val text: String,
    /** Audio time since engine start, for measuring end-to-end latency. */
    val startMs: Long,
    val endMs: Long,
    /** 0..1, or null if the engine can't provide it. Used to fade low-confidence words. */
    val confidence: Float? = null,
)

data class EngineConfig(
    val language: String = "he-IL",
    val threads: Int = 4,            // 6 and 8 were slower on Snapdragon 695 (slow cores); 2 and 3 untested
    val chunkMs: Int = 1120,         // Nemotron: 560 as option. Ignored by Whisper.
)
