# Phase 0: Whisper Phone Test Results

*September 26, 2026 · Whisper fallback track (Plan B)*
*Phone: Poco X4 Pro 5G (Snapdragon 695), CPU, whisper.cpp in Termux · Model: stock multilingual Whisper `base`, `q8_0`*

---

## 1. Key findings

1. **Stock Whisper `base` hallucinates on every kind of non-speech** (digital silence, recorded room silence, pink noise, TV noise): random Hebrew and English words, even Chinese/Korean/Polish characters. It is also slow there: the decoder ran 700–1,100 steps (6–10 s per run) instead of ~35 steps (0.74 s) on real speech.
2. **Silero VAD fixes it completely.** With VAD, all four non-speech clips produce no text, in ~0.25 s per run. Clean speech and noisy speech (mild and hard) still transcribe correctly with VAD on. **VAD is mandatory for Whisper, not optional.**
3. **VAD passes quiet speech from 2.5 m.** This is correct behavior: VAD detects speech, not who the speech is addressed to. Whether to caption background speech is a product question for testers.
4. **Distant speech (2.5 m) produced fluent-looking but likely wrong Hebrew.** This is the most dangerous error type for deaf users, since they can't detect it. *(Still to confirm: what was actually said.)*
5. **Code-switching:** English words ("deadline", "report") came out in Hebrew letters, with some word errors.
6. **Sustained use:** ~970 ms per run at the start, rising to a stable plateau of ~1,230 ms after ~5 minutes (+27%). Battery dropped 65% → 62% in ~11 minutes of non-stop work (roughly 15–20% per hour, worst case).
7. **Realistic sustained speed for planning: ~1.2–1.3 s per update** (base, ~7 s buffer).

---

## 2. Best phone configuration

```bash
./build/bin/whisper-cli -m models/ggml-base-q8_0.bin -f CLIP.wav -l he -t 6 -ac 512 -bs 1 -bo 1 \
  --vad -vm models/ggml-silero-v5.1.2.bin
```

| Flag | Meaning |
|---|---|
| `-t 6` | 6 threads (fastest on this phone) |
| `-ac 512` | Encoder covers ~10.2 s of audio (50 positions = 1 s) |
| `-bs 1 -bo 1` | Greedy decoding, single attempt |
| `q8_0` | 8-bit model (faster than 5-bit on this phone) |
| `--vad` | Silero VAD, always on |

Cap the uncommitted buffer at ~6–8 s so `-ac 512` always fits.

---

## 3. Results log

### Speed tests (7-second clean Hebrew clip, no VAD)

| Settings | Time | Output |
|---|---|---|
| `-t 2` | 5.58 s | baseline |
| `-t 4` | 5.36 s | same |
| `-t 6` | 4.45 s | same |
| `-t 8` | 4.69 s | same |
| `-t 6 -ac 512` | 2.31 s | same |
| `-t 6 -ac 768` | 2.85 s | slightly better |
| `-t 6 -ac 512 -bs 1 -bo 1` | 1.35 s | same |
| + `q8_0` | **0.74 s** | same |
| + `q5_0` (instead of q8_0) | 1.15 s | same |
| 12-second clip, q8_0, `-ac 768`, greedy | 1.50 s | Time grows faster than audio length |

### Non-speech clips (hallucination check)

| Clip | Flags | Encode | Decode | Total | Output |
|---|---|---|---|---|---|
| Digital silence (10 s) | — | 376 ms | 4,577 ms (727 runs) | 6,271 ms | Lots of invented Hebrew/English ❌ |
| Recorded room silence (10 s) | — | 363 ms | 5,319 ms (767 runs) | 7,312 ms | Invented words ❌ |
| Pink noise (10 s) | — | 372 ms | 4,615 ms (724 runs) | 6,207 ms | 1 Hebrew letter + 1 English word ❌ |
| TV noise (10 s) | — | 364 ms | 7,383 ms (1,101 runs) | 10,345 ms | Lots of invented text ❌ |
| Digital silence | VAD | 0 | 0 | 260 ms | Nothing ✅ |
| Recorded room silence | VAD | 0 | 0 | 260 ms | Nothing ✅ |
| Pink noise | VAD | 0 | 0 | 252 ms | Nothing ✅ |
| TV noise | VAD | 0 | 0 | 255 ms | Nothing ✅ |

Hallucination signs worth using as filters: segments ending after the audio ends (up to 17 s on 10 s clips), and mixed foreign scripts.

### Speech clips

| Clip | Flags | Total | Output |
|---|---|---|---|
| Clean 7 s clip | VAD | ~970 ms | Same transcript as without VAD ✅ |
| Noisy mild / noisy hard | VAD | — | Correct ✅ |
| Code-switching | VAD | — | English words in Hebrew letters, some word errors |
| Quiet speech from 2.5 m | VAD | 2,671 ms (encode 361, decode 1,289 / 160 runs) | Fluent-looking but likely wrong Hebrew ⚠️ |

### Heat and battery (clean 7 s clip, base q8_0, `-ac 512`, VAD)

| Test | Result |
|---|---|
| 20 runs | 965–998 ms, no creep |
| 600 runs (~11 min) | ~970 ms → ~1,050 ms (after ~40 s) → ~1,100 ms (~2 min) → **~1,230 ms plateau** (after ~5 min, to the end). Occasional one-off spikes (1,382 ms, 1,701 ms). Battery 65% → 62%. |

Note: each test run reloads both models from storage (~0.23 s). An app keeps them loaded, so real per-update time will be somewhat lower.

---

## 4. Checklist status

- [x] Silence-only clips (digital and recorded)
- [x] Noise-only clips (pink noise, TV noise)
- [x] VAD test
- [x] Noisy speech clips
- [x] Code-switching clip
- [x] Heat test (20 runs and 600 runs) + battery
- [x] Encode vs. decode time recorded
- [ ] 3-second clip with `-ac 256` *(skipped)*
- [ ] `tiny` model *(skipped)*
- [ ] Hebrew-tuned model *(superseded: Nemotron became Plan A; only if Plan B is activated)*
- [ ] Write down what was actually said in the 2.5 m clip; run it with `-pc` (confidence)

Scripts used: `phone-bench/whisper/`.
