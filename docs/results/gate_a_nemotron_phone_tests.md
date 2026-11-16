# Gate A: Nemotron Phone Test Results

*October 1, 2026 · Plan A (Nemotron) · Gate A*
*Phone: Poco X4 Pro 5G (Snapdragon 695: 2 fast + 6 slow cores, 7.3 GiB RAM), CPU, NeMo-Speech.cpp 0.1.0 in Termux*
*Model: `nemotron-3.5-asr-streaming-0.6b.q8_0.gguf` (zero-shot, not fine-tuned)*

---

## 1. Key findings

1. **Nemotron runs live on the phone.** Streaming RTF ~0.43 when cool.
2. **When hot, it settles on a stable plateau**: average RTF ~0.59, worst ~0.68, over ~14 minutes of non-stop speech (1120 ms chunks). It does not keep climbing.
3. **Gate A: conditional pass.** Misses the strict "≤ 0.5 when hot" target, but runs live with ~30–40% spare capacity. Real conversations (pauses + VAD) will be lighter. **Nemotron becomes Plan A.**
4. **No invented text on silence or noise, even without VAD** (digital silence, room silence, pink noise, TV noise). Whisper invented text on all four.
5. **Hebrew zero-shot** on the clean 7 s clip: few mistakes.
6. **4 threads is best.** 6 and 8 threads were slower: extra threads land on the slow cores and spend their time waiting (system time 17 s → 1m42 s → 4m20 s).
7. **Larger chunks are lighter:** 1120 ms < 560 ms < 320 ms in processing time. Whole-file mode is slowest (RTF ~1.2) but isn't used in the app.
8. **Charging slows the phone ~35%** (heat). Always test unplugged and cooled down.
9. **Battery: ~17% per hour** of non-stop speech (similar to Whisper's 15–20%).
10. **Startup costs ~4 s** (model load + warmup), once per app start.

---

## 2. Results

RTF = (processing time − ~4 s startup) ÷ audio length. `long.wav` = 66 s (jfk.wav × 6).

### Speed tests (unplugged, 4 threads)

| Test | Audio | Time | RTF |
|---|---|---|---|
| `tiny.wav` (startup cost) | 1 s | 4.2 s | — |
| `jfk.wav`, whole-file | 11 s | 11.1 s | — |
| `long.wav`, whole-file | 66 s | 81.9 s | ~1.2 ❌ |
| `long.wav`, stream 320 ms (R=3) | 66 s | 43.3 s | ~0.60 |
| `long.wav`, stream 560 ms (R=6) | 66 s | 40.1 s | ~0.55 |
| `clip7s.wav` Hebrew `he-IL`, stream 560 ms | 7 s | 7.3 s | (mostly startup) |

### Thread sweep (stream 320 ms; phone was warm/charging, so times are high)

| Threads | Real | User | Sys |
|---|---|---|---|
| 4 | 54.8 s | 3m13 s | 17 s |
| 6 | 56.8 s | 3m49 s | 1m42 s |
| 8 | 76.0 s | 5m02 s | 4m21 s |

After unplugging and cooling: the same 4-thread test was back to ~40 s.

### Heat test 1: 10 runs, 560 ms (R=6), battery 84% → 82%

| Run | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Time (s) | 34.5 | 33.1 | 36.4 | 37.5 | 38.8 | 38.9 | 40.5 | 42.7 | 52.1 | 47.8 |
| RTF | 0.46 | 0.44 | 0.49 | 0.51 | 0.53 | 0.53 | 0.55 | 0.59 | 0.73 | 0.66 |

### Heat test 2: 20 runs, 1120 ms (R=13), ~14 min, battery 81% → 77%

| Run | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Time (s) | 32.4 | 31.6 | 31.5 | 32.4 | 34.7 | 39.8 | 42.9 | 39.6 | 38.2 | 49.1 |
| RTF | 0.43 | 0.42 | 0.42 | 0.43 | 0.47 | 0.54 | 0.59 | 0.54 | 0.52 | 0.68 |

| Run | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|
| Time (s) | 46.9 | 39.6 | 38.9 | 49.2 | 40.3 | 43.4 | 48.5 | 40.7 | 48.6 | 39.9 |
| RTF | 0.65 | 0.54 | 0.53 | 0.68 | 0.55 | 0.60 | 0.67 | 0.56 | 0.68 | 0.54 |

Runs 6–20: average 43.0 s (RTF ~0.59), range 38.2–49.2 s. Swings = the phone stepping its speed up and down for heat; no upward trend.

### Hallucination check (`he-IL`, stream 1120 ms, no VAD)

| Clip | Output |
|---|---|
| Digital silence (10 s) | Nothing ✅ |
| Recorded room silence (10 s) | Nothing ✅ |
| Pink noise (10 s) | Nothing ✅ |
| TV noise (10 s) | Nothing ✅ |

Rerun these after every fine-tuning run: training can change this behavior.

---

## 3. Gate A decision

| Gate | Result | Decision |
|---|---|---|
| A: phone speed | Cool RTF ~0.43; hot plateau avg ~0.59, worst ~0.68; no hallucinations | **Conditional pass → Nemotron is Plan A, Whisper is Plan B** |

App defaults from these tests: **1120 ms chunks** (560 ms option), **4 threads**, model loaded once, VAD to pause during silence.

---

## 4. Still to test on the phone (optional)

- [ ] Noisy clips (`noisy_mild`, `noisy_hard`), `2.5m.wav`, and the code-switching clip with `he-IL` (compare with Whisper)
- [ ] After fine-tuning: the same speed and hallucination tests with the Hebrew model

---

## 5. Termux build reference (NeMo-Speech.cpp)

The full build is scripted in `phone-bench/nemotron/build_termux.sh`.

### Problems hit and fixes

| Problem | Fix |
|---|---|
| `required submodules are not initialized: llama.cpp` | `git submodule update --init --recursive llama.cpp` |
| `Could not find SENTENCEPIECE_LIB` | Build SentencePiece from source |
| Build error `no member named 'partial_ordering' in namespace 'std'` | Recent SentencePiece's header includes system abseil (needs C++20; NeMo-Speech.cpp uses C++17). Use **SentencePiece v0.2.0** |
| SentencePiece v0.2.0: `Compatibility with CMake < 3.5 has been removed` | Add `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` |
| `cannot locate symbol "__android_log_write"` | Link Android's log library: `-DCMAKE_SHARED_LINKER_FLAGS="-llog" -DCMAKE_EXE_LINKER_FLAGS="-llog"` |
| Threads hardcoded to 4 (`src/runtime/ggml/session.cpp:911`) | Patched to read `NS_THREADS` |

CPU features detected by ggml: dotprod ✅, FP16 arithmetic ✅, i8mm ❌, SVE ❌.

### Useful options

| Option | Meaning |
|---|---|
| `--stream` | Cache-aware streaming from a WAV file (as in the app) |
| `--asr.streaming.rnnt_right_context R` | Chunk = (R + 1) × 80 ms: 3 → 320 ms, 6 → 560 ms, 13 → 1120 ms |
| `-l he-IL` / `-l en-US` | Language prompt |
| `--live` | Microphone transcription |
| `--vad-model PATH` | Silero VAD |
| `NS_THREADS=N` | Thread count (after the patch) |
| `NEMO_SPEECH_TIMING=1` | Per-stage timing output |
| `ns help transcribe` | All options |

Testing rules: phone unplugged, cooled down ~10 min, screen off between runs if possible; record battery % before and after.
