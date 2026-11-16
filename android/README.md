# Android app (minimal app in Phases 1–2, full prototype in Phase 3 — planned)

Start here: `SpeechEngine.kt` (the interface both engines implement, with structured `CaptionUpdate` events).
Design: `docs/ARCHITECTURE.md` §3.

**First milestone (parallel track, `docs/PLAN.md` §7):** a minimal app with the *zero-shot* model:
- JNI + AudioRecord + VAD + one text view
- microphone foreground service
- measure end-to-end latency, screen-on heat/battery, peak RAM
- find out what the C ABI exposes (timestamps, token probabilities)

If NeMo-Speech.cpp won't integrate, try sherpa-onnx with the same model (D-014).

Planned structure:
```
android/
  app/                    Compose UI, caption state, settings
  asr/                    SpeechEngine, NemotronEngine (JNI → NeMo-Speech.cpp C ABI), WhisperEngine (JNI → whisper.cpp)
  audio/                  AudioRecord 16 kHz mono, ring buffer, Silero VAD (300 ms pre-roll)
  glasses/                BLE caption diffs (Phase 5)
```

Checklist (from `docs/PLAN.md` Phase 3):
- [ ] Gradle project + NDK/CMake build of NeMo-Speech.cpp for arm64-v8a, linking `liblog`
- [ ] NemotronEngine: model loaded once, 4 threads, 1120 ms chunks (560 option)
- [ ] Audio capture + VAD gating (300 ms before speech, ~0.5–1 s after speech), backpressure indicator
- [ ] Microphone foreground service + battery-optimization exemption (long sessions)
- [ ] Caption UI: stable text normal, unstable tail grey/italic, smooth word reveal, mixed Hebrew/English/digits don't reorder
- [ ] Model downloaded on first run (~650 MB), not bundled in the APK
- [ ] 10–15 min real conversation test: latency, heat, battery
- [ ] 2–3 deaf testers
