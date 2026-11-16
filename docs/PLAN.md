# Hebrew Live Captions for Deaf Users: Project Plan v3

*Last updated: October 26, 2026 (critical review: gates, Plan B scope, Android track moved earlier; see D-011 to D-016) · Replaces plan v1 and plan v2 (archived in the claude.ai Project)*

> **How to use this file:** This is the source of truth for "what's next". Results go in `docs/results/`, decisions in `docs/DECISIONS.md`. Update the checkboxes, the gate table, and the results log as you go (Claude Code does this too; see `CLAUDE.md`).

---

## 1. Project summary

**Goal:** A live Hebrew speech-to-text system for deaf and hard-of-hearing people, running **fully on-device** on an Android phone (later with smart glasses as a display), with low latency.

**Plan A (primary): NVIDIA Nemotron 3.5 ASR Streaming 0.6B, fine-tuned for Hebrew.**
- True streaming (cache-aware FastConformer-RNNT): each audio chunk is processed once, no re-transcribing.
- Chunk size chosen at runtime (80 / 160 / 320 / 560 / 1120 ms).
- Hebrew is already in the tokenizer: fine-tune directly.
- Outputs punctuation.
- **Gate A passed (conditional) on the test phone, October 1, 2026.** Runs live, stable when hot, no invented text on silence or noise.
- Local runtime: NeMo-Speech.cpp (C++/ggml) with GGUF `q8_0` models.

**Plan B (fallback): Whisper `base`, Hebrew-tuned, + re-transcribe-and-commit loop with whisper.cpp.**
- Proven on the test phone (~1.2–1.3 s per update when hot), but needs mandatory VAD (hallucinates on all non-speech) and its own Hebrew fine-tune (no good small Hebrew Whisper exists).
- Activated only if a Plan B trigger happens (section 3). **It can't fix an accuracy failure** (a 74M model won't beat a fine-tuned 0.6B one), and it isn't clearly lighter on the CPU. So in practice it covers the license trigger, and the integration trigger only if sherpa-onnx also fails (D-016, proposed).

---

## 2. Status at a glance

| Item | Status |
|---|---|
| Whisper phone tests (Phase 0) | ✅ Done (Sept 26) |
| Hebrew leaderboard reviewed | ✅ Done; Nemotron zero-shot baseline found |
| Nemotron Gate A (phone speed) | ✅ **Conditional pass** (Oct 1) |
| Nemotron silence/noise check (phone) | ✅ No invented text, even without VAD |
| Repo scaffolded | 🟡 (Oct 26): docs + scaffolding written; `git init` + first push still to do |
| PC setup (Fedora) | ⬜ **Next step** |
| Phone re-tests for Gate A′ (Hebrew, screen on, 60 min) | ⬜ |
| User research: target user + first contacts | ⬜ (now Phase 1) |
| Evaluation harness + baselines | ⬜ |
| Hebrew fine-tuning pilot → Gate B | ⬜ |
| Android minimal app (zero-shot model) | ⬜ (parallel track, Phases 1–2) |
| Android prototype with testers → Gate C | ⬜ |

---

## 3. Decision gates

### Gate A: Can Nemotron run live on the phone? → ✅ Conditional pass

- **Result:** RTF ~0.43 when cool; stable plateau when hot (average ~0.59, worst ~0.68 over 14 min of non-stop speech, 1120 ms chunks). Never close to 1.0.
- **Why "conditional":** misses the strict "RTF ≤ 0.5 when hot" target, so headroom is smaller than ideal. Acceptable because real conversations have pauses (VAD stops processing during silence), and startup cost is paid once.
- **Optimization paths for later (not blockers):** q4 quantization, thread count/affinity, distilled smaller streaming student, NPU, newer phones.
- **Caveat (D-012):** the ≤ 0.5 target was relaxed after the measurement. The test also ran in easier-than-real conditions:
  - English audio (`jfk.wav` × 6, `-l en-US`)
  - screen off between runs
  - 14 minutes only
  - the VAD saving assumed, not measured

  → Gate A′.

### Gate A′: Does it still run live in realistic conditions? → ⬜ Phase 1 (Termux), again in Phase 3 (app)

- **Test:**
  - ≥ 60 s of Hebrew audio with `-l he-IL`, looped for 60 min
  - screen on
  - threads 2 / 3 / 4, phone cool and unplugged
  - q8_0 vs q4_0
  - peak RAM recorded
- **Pass:** hot-plateau RTF stays below ~0.8 with no upward trend over 60 min. Record battery %/h with the screen on.
- **If it fails:** levers in order: q4 → 1120 ms chunks → threads/affinity → newer phone → smaller student.

### Gate B: Is fine-tuned Nemotron good in Hebrew? → ⬜

The leaderboard metric is **WER** (D-011). Always report CER next to it.

- **Test:** fine-tune on 100–200 h of Hebrew. Also train a small 50 h run, so the two points show whether more data helps. Evaluate in streaming mode at 560 and 1120 ms on:
  - eval-d1
  - eval-whatsapp
  - the real-world set (Phase 1)
  - the other leaderboard sets
  - the hallucination suite
- **Pass, all of:**
  - eval-d1 WER ≤ ~0.08 (provisional; copied from the ivrit.ai models)
  - eval-whatsapp WER ≤ **TBD**
  - real-world set WER ≤ **TBD** (set from user research and the Live Transcribe benchmark)
  - hallucination suite: no output at all
  - code-switching acceptable on own clips

  eval-d1 alone doesn't count: it's one ~2 h episode and the set closest to the training data.
- **If it fails:** scale up data (Phase 4) up to a **limit: TBD** (time/data budget, open decision). Only scale if the 50 h → 200 h step showed a clear improvement. Past the limit → trigger 1.

### Gate C: Does it help deaf users? → ⬜ Phase 3

- **Test:** 2–3 (later more) deaf testers use the prototype in a real conversation. Compare with Google Live Transcribe on the same conversation.
- **Measures (decide with the user research):**
  - can they follow the conversation (a short comprehension check)
  - would they use it again
  - is the delay OK
  - did they notice wrong text
- **Pass:** testers prefer it, or find it usable, in at least one situation they care about.

| Gate | Result | Date | Decision |
|---|---|---|---|
| A: phone speed | Cool RTF ~0.43; hot plateau ~0.59 avg / 0.68 worst; no hallucinations | Oct 1, 2026 | **Conditional pass → Nemotron is Plan A** |
| A′: realistic phone conditions | | | |
| B: Hebrew accuracy | | | |
| C: user value | | | |

### Triggers and their fallbacks

Each trigger has its own fallback. Whisper `base` (Plan B) is the answer to only some of them (D-016, proposed).

| # | Trigger | Fallback |
|---|---|---|
| 1 | Gate B fails after scaling up to the limit | More/better data, distillation from the ivrit.ai teacher, rethink the model. **Not** Whisper `base`: it would be less accurate. |
| 2 | NeMo-Speech.cpp can't be integrated into the Android app | **Keep the model, swap the runtime:** sherpa-onnx (Android SDK; runs Nemotron 3.5 streaming on desktop, chunks 80 ms–1.12 s). Unverified on Android with a fine-tuned model. Whisper only if that also fails. |
| 3 | Testers find heat, battery, or delay unacceptable | Gate A′ levers: q4, chunk size, threads, newer phone, smaller student. Whisper `base` isn't clearly lighter (re-transcribes ~7 s every ~1.2 s). |
| 4 | The OpenMDW-1.1 license blocks shipping | Switch model: Whisper `base` (Plan B) or another streaming model. |

---

## 4. Key decisions

| Decision | Choice | Notes |
|---|---|---|
| Platform | **Android** | Apple dropped. |
| Plan A engine | **Nemotron 3.5 ASR Streaming 0.6B** | Gate A passed (conditional). |
| Plan B engine | **Whisper `base`, Hebrew-tuned** | Needs own fine-tune; only for the triggers it actually answers (§3, D-016 proposed). |
| Whisper `base` fine-tune | **Dropped for now** | Only if Plan B is activated. |
| Phone runtime | **NeMo-Speech.cpp** (sherpa-onnx as fallback) | D-014 (proposed). Pin the commit hash. |
| Quantization | **q8_0, provisional** | Chosen from a Whisper result; test q4_0 for Nemotron (D-015). |
| Default chunk size | **1120 ms** (560 ms as option) | 1120 ms is lighter; testers decide if 560 ms is worth it. |
| CPU threads | **4** (2 and 3 untested) | 6 and 8 were slower on this phone; re-test cool + unplugged in Gate A′. |
| VAD | **Still used** (Silero) | For battery during silence; Nemotron itself didn't hallucinate. Keep ~0.5–1 s of audio after speech ends (D-013). The saving in real conversations is unmeasured. |
| Testing rule | **Phone unplugged, cooled down** | Charging made runs ~35% slower. |
| Chip (SoC) | **Decide later** | Train chip-independent; convert per chip at the end. |
| Languages | **Hebrew + English** (code-switching) | 10–20% English and code-switched examples in training data. |
| App design | **Swappable `SpeechEngine` interface** | Both engines plug into the same app. |
| Test phone | **Poco X4 Pro 5G** (Snapdragon 695) | Typical mid-range phone. |
| Training PC | **Own PC, Fedora, RTX 5060 Ti 16 GB** | Native Linux (no WSL2). Rent cloud GPUs only for big jobs. |
| Evaluation | **ivrit.ai leaderboard test sets + code** | Comparable numbers; no harness from scratch. Metric for comparison is WER, CER always reported next to it (D-011). Normalization from `evaluate_model.py` in `ivrit-ai/asr-training`. |

---

## 5. Phone configurations

### Plan A: Nemotron (NeMo-Speech.cpp in Termux)

```bash
ns transcribe CLIP.wav -l he-IL --stream --asr.streaming.rnnt_right_context 13
```

| Setting | Meaning |
|---|---|
| `--stream` | Cache-aware streaming (as in the app) |
| `rnnt_right_context R` | Chunk = (R + 1) × 80 ms: R=1 → 160 ms, 3 → 320 ms, 6 → 560 ms, 13 → 1120 ms |
| `-l he-IL` | Language prompt |
| Threads | Hardcoded 4 in `session.cpp:911`; patched to read `NS_THREADS` (4 is best) |
| `NEMO_SPEECH_TIMING=1` | Prints per-stage timing |

Build details and fixes: see `docs/results/gate_a_nemotron_phone_tests.md`.

### Plan B: Whisper (whisper.cpp in Termux)

```bash
./build/bin/whisper-cli -m models/ggml-base-q8_0.bin -f CLIP.wav -l he -t 6 -ac 512 -bs 1 -bo 1 \
  --vad -vm models/ggml-silero-v5.1.2.bin
```

VAD mandatory. Cap the uncommitted buffer at ~6–8 s. Details: `docs/results/phase0_phone_tests_summary.md`.

---

## 6. Results log

### 6.1 Nemotron Gate A (Poco X4 Pro, CPU, q8_0, 4 threads, unplugged)

RTF = (processing time − ~4 s startup) ÷ audio length. Lower is better; below 1.0 runs live.

| Test | Result |
|---|---|
| Startup (model load + warmup) | ~4 s, once per app start |
| 66 s clip, whole-file | 81.9 s (RTF ~1.2; not used in the app) |
| 66 s clip, stream 320 ms | 43.3 s → RTF ~0.60 |
| 66 s clip, stream 560 ms | 40.1 s → RTF ~0.55 |
| 10-run heat, 560 ms | RTF 0.44–0.49 cool → 0.59–0.73 by runs 8–10; battery 84% → 82% |
| **20-run heat, 1120 ms (~14 min)** | **RTF ~0.43 cool → stable plateau from run 6: avg ~0.59, worst ~0.68**; battery 81% → 77% (~17%/h) |
| Threads (320 ms, phone warm) | 4: 54.8 s · 6: 56.8 s · 8: 76.0 s (system time 17 s → 1m42 s → 4m20 s) |
| Charging effect | Same test: ~55 s while charging/hot vs ~40 s unplugged and cool |
| Silence/noise clips (no VAD) | Digital silence, room silence, pink noise, TV noise: **no text** ✅ |
| Hebrew zero-shot (clean 7 s clip) | Few mistakes ✅ |

### 6.2 Hebrew leaderboard baselines (ivrit.ai, **WER**, lower is better)

Source: `benchmark.csv` in huggingface.co/spaces/ivrit-ai/hebrew-transcription-leaderboard. The leaderboard scores WER with jiwer (`evaluate_model.py` in `ivrit-ai/asr-training`).

Set sizes:
- eval-d1: ~2 h, **one podcast episode**
- eval-whatsapp: ~1:10 h
- SASpeech: ~4 h
- FLEURS-he: ~2 h
- Kan: ~1.7 h
- The leaderboard also has Common Voice 17 he (~2 h), not copied here yet.

| Model | eval-d1 | eval-whatsapp | SASpeech | FLEURS-he | Kan |
|---|---|---|---|---|---|
| **Nemotron 3.5 streaming 0.6B (zero-shot)** | **0.171** | **0.365** | **0.242** | **0.460** | **0.542** |
| ivrit-ai whisper-large-v3-turbo (20250513) | 0.053 | 0.071 | 0.066 | 0.181 | 0.082 |
| ivrit-ai whisper-large-v3 (20250513) | 0.051 | 0.072 | 0.064 | 0.174 | 0.081 |
| Soniox (commercial, best) | 0.048 | 0.090 | 0.065 | 0.177 | 0.114 |
| Amazon Transcribe batch | 0.066 | 0.104 | 0.085 | 0.230 | 0.090 |
| Amazon Transcribe stream | 0.079 | 0.129 | 0.090 | 0.287 | 0.131 |
| Google Speech (cloud API) | 0.211 | 0.352 | 0.189 | 0.385 | 0.292 |

Takeaways:
- Nemotron zero-shot is a real starting point on conversational speech (eval-d1), weak elsewhere. Fine-tuning goal: close the gap to ~0.05–0.08 WER. eval-whatsapp (phone mics, volunteers) is closer to the real use case than eval-d1.
- Best teacher for pseudo-labeling: **ivrit-ai whisper-large-v3-turbo-ct2-20250513** (near large-v3 quality, faster).
- Streaming costs some accuracy (Amazon batch vs stream). Expect a few points' gap vs. batch models.
- The leaderboard doesn't state Nemotron's chunk size/mode (maybe in its `machine.json`).
- ivrit.ai publishes only large Whisper models. Community small Hebrew Whispers are weak (small ~0.38 WER, tiny ~0.56 on ivrit-ai eval).

### 6.3 Whisper Plan B (summary; full data in `docs/results/phase0_phone_tests_summary.md`)

| Test | Result |
|---|---|
| Best speed (7 s clip, base q8_0) | 0.74 s cold; ~1.23 s plateau when hot |
| Non-speech without VAD | Invents text on all 4 clips ❌ |
| Non-speech with VAD | No text ✅ |
| Distant speech (2.5 m) | Fluent-looking but likely wrong ⚠️ |
| Code-switching | English words in Hebrew letters, some errors |
| Battery | ~15–20%/h non-stop |

---

## 7. Next steps

### Phase 1: PC setup, evaluation harness, baselines (current step)

**Fedora environment** (step-by-step in `docs/GETTING_STARTED.md`)
- [ ] NVIDIA driver from RPM Fusion: `sudo dnf install akmod-nvidia xorg-x11-drv-nvidia-cuda`, reboot, `nvidia-smi` shows the RTX 5060 Ti and driver 570+ (Blackwell needs the open kernel modules; enroll the key if Secure Boot is on)
  - [ ] Check that the **open** kernel module is the one loaded (`modinfo nvidia | grep -i license` should show "Dual MIT/GPL"; unverified what RPM Fusion builds by default, see its NVIDIA howto)
- [ ] Python 3.11 env with uv: `uv venv ~/asr-env --python 3.11`
- [ ] PyTorch for Blackwell: `uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128`; check `torch.cuda.get_device_capability()` → `(12, 0)`
- [ ] NVIDIA NeMo: `uv pip install "nemo_toolkit[asr]"`, then **re-run the torch check**: installing NeMo can replace the cu128 torch build
- [ ] Record the installed NeMo, torch and driver versions in the results log
- [ ] Storage (D-019): 500 GB NVMe + optional 500 GB–1 TB SATA. Set `HF_HOME` (and `HELICA_DATA`) before any dataset download; never download all of `audio-v2`

**Phone re-tests (Gate A′, Termux; no PC needed)**
- [ ] Pin NeMo-Speech.cpp: `NS_REF=<commit> bash phone-bench/nemotron/build_termux.sh` (the script saves the commit to `~/ns_build_commit.txt`)
- [ ] Make a ≥ 60 s Hebrew clip (e.g. a podcast segment) and heat-test it in Hebrew: `bash heat_test.sh 13 he-IL clips/hebrew_long.wav`
- [ ] 60-minute run with the **screen on** (as in the app); battery %/h
- [ ] Thread sweep 2 / 3 / 4, phone cool and unplugged
- [ ] q4_0 vs q8_0: speed, and output text on the own clips
- [ ] Peak RAM (e.g. `/usr/bin/time -v` if available in Termux, or `top` while running)
- [ ] Zero-shot `he-IL` on noisy clips, the 2.5 m clip and the code-switching clip (moved here from "optional": tells you early whether far-field data is needed)

**Evaluation harness** (`eval/`)
- [ ] Copy the normalization from `evaluate_model.py` in `github.com/ivrit-ai/asr-training`; that's what the leaderboard links to, not the Space's `src/`
- [ ] Test sets: ivrit-ai/eval-d1, ivrit-ai/eval-whatsapp, SASpeech, FLEURS-he, Common Voice 17 he, hebrew_speech_kan
- [ ] Own clips: clean, noisy, 2.5 m, code-switching
- [ ] Write down the exact reference text for each own clip (including what was said in the 2.5 m clip)
- [ ] **Real-world test set:** 30–60 min of realistic conversations, recorded with everyone's consent and hand-transcribed (teacher draft, then corrected by hand):
  - 1–3 m away, several speakers, overlapping speech
  - phone on a table, background noise
  - Part of Gate B.
- [ ] **Hallucination suite, grown to 30–60 min:**
  - music (incl. lyrics in other languages), café babble, kitchen, street, wind
  - phone rubbing in a pocket, cough / laugh / breathing
  - Report characters per minute. Test both the model alone (no VAD) and the system with VAD, as in the app. Check each noise corpus's license.
- [ ] Insertions inside speech: text during pauses or after speech ends, on the speech sets
- [ ] Per-clip output logging (download Nemotron's per-sample CSVs from the leaderboard to study its Hebrew errors)

**Baselines on the PC**
- [ ] Nemotron zero-shot `he-IL`: reproduce the leaderboard numbers (confirms the harness), then at 560 and 1120 ms chunks
- [ ] ivrit-ai whisper-large-v3-turbo on one set only, to double-check the harness (the rest are on the leaderboard)

**Before training**
- [ ] Read NVIDIA's guide: "How to Fine-Tune Nemotron 3.5 ASR for Your Language, Domain, or Accent"
- [ ] Licenses: OpenMDW-1.1 (fine-tuning, distillation, shipping) and each ivrit.ai dataset (the "ivrit.ai License", gated); fill `docs/DECISIONS.md` → Licenses register
- [ ] **Tokenizer check:** tokenize ~1,000 ivrit.ai sentences. Measure tokens per word, `<unk>` rate, and which characters are unknown (niqqud, geresh, maqaf, final letters). Hebrew tokens per second of speech vs English tells you whether decoding will be slower in Hebrew.
- [ ] **Text-convention audit** of the ivrit.ai transcripts:
  - English words: Latin or Hebrew letters?
  - numbers: digits or words?
  - punctuation density
  - niqqud

  Adopt their convention for training unless there's a strong reason not to (re-labeling is expensive). Numbers can be converted at display time. Latin vs Hebrew letters can't be changed after training, so decide it here.
- [ ] English / code-switched data: pick a source and check its license. There's no public Hebrew-English code-switched set; ivrit.ai podcasts contain natural code-switching.
- [ ] **200-step memory smoke run** with `training/configs/pilot_16gb.yaml` values: peak VRAM, steps/s, wall-clock estimate for the full run

**Exit criteria:**
- training runs on the PC within 16 GB (smoke run done)
- the harness reproduces the leaderboard number
- baselines recorded
- Gate A′ decided
- target user defined (§8)

### Phase 2: Hebrew fine-tuning pilot → Gate B

- [x] Pilot clips selected: crowd-transcribe-v5, human-transcribed (D-020 to D-022). `training/scripts/select_crowd_v5.py` → 182,586 train clips / 264.2 h, 17,964 validation clips / 26.1 h
- [ ] Pilot Stage 2 (Fedora PC): audio of the selected clips → 16 kHz mono FLAC + NeMo manifest (`audio_filepath`, `duration`, `text`). Script not written yet.
- [ ] **Exclude the eval-d1 podcast series** (not just the episode: same hosts, same topics) from training data. Pilot: `exclude_sources` prefixes in `eval/test_sets.yaml`, applied by `select_crowd_v5.py` (episode-level; the whole Mindset series is in crowd-transcribe `test`, so train has none, D-020). Scale-up with `build_manifests.py`: pass `--source-col`, otherwise the guard can't work (the script refuses).
- [ ] Check eval transcripts against training transcripts for duplicates (text hash)
- [ ] Sample for diversity: pilot = the whole filtered crowd-transcribe train split. Scale-up: shuffled (`--seed`), capped per source (`--max-hours-per-source`) in `build_manifests.py`
- [ ] Validation set held out **by episode**, never random utterances. Pilot: crowd-transcribe `test` split (0 shared episodes with train, D-021) minus the eval-d1 rows (D-022). Scale-up: `--val-frac` in `build_manifests.py`
- [ ] Add 10–20% English and code-switched examples (source chosen in Phase 1)
- [ ] 16 GB settings from the smoke run: bf16, fused joint/loss, small batch + gradient accumulation, max audio duration per sample
- [ ] Keep the base model's multi-lookahead setup (all `att_context_size` values); pick checkpoints by streaming WER at 1120 ms
- [ ] Two data points: 50 h and 100–200 h, to see whether more data still helps
- [ ] Evaluate at 560 and 1120 ms chunks (plus 160/320 ms for the curve), WER **and** CER
- [ ] Check the hallucination suite (first!), noisy and distant speech, code-switching, punctuation
- [ ] **Confidence check:** do low-confidence tokens line up with errors on the eval sets? (This is the defence against fluent-but-wrong text on screen.)
- [ ] Export to GGUF `q8_0` (`convert_model.py`), parity check on ~200 eval utterances (exact-match rate + CER difference), run on the phone, confirm the output matches the PC
- [ ] **Decide Gate B**

### Parallel track: Android minimal app (during Phases 1–2, while the GPU trains)

Follows D-002: these risks don't depend on fine-tuning, so retire them early.

- [ ] Gradle + NDK build of NeMo-Speech.cpp (arm64-v8a, `-llog`), JNI wrapper, zero-shot model
- [ ] AudioRecord 16 kHz mono → Silero VAD → engine → one text view
- [ ] Microphone foreground service + battery-optimization exemption, so long sessions don't get killed
- [ ] Measure end-to-end latency (word spoken → on screen), heat and battery with the screen on, peak RAM
- [ ] Find out what the C ABI exposes: token timestamps? token probabilities (confidence)?
- [ ] If the integration fails: try sherpa-onnx with the same model (trigger 2 fallback)
- [ ] Show it to the first deaf contacts (feeds §8)

### Phase 3: Android prototype

Builds on the minimal app from the parallel track.

- [ ] `SpeechEngine` interface with structured `CaptionUpdate` events (`android/SpeechEngine.kt`)
- [ ] **NemotronEngine:** NeMo-Speech.cpp native SDK (C ABI) via JNI; streaming at 1120 ms (560 ms option); 4 threads; model loaded once at app start
- [ ] Link `liblog` in the Android build (needed in Termux too)
- [ ] Microphone: 16 kHz mono (AudioRecord), foreground service
- [ ] VAD (Silero): pause processing during silence (battery), keeping ~0.5–1 s of audio after speech ends. Compare punctuation with and without.
- [ ] Display: Nemotron text is append-only, so only the last incomplete word is grey. Fade low-confidence words if the confidence check (Phase 2) worked.
- [ ] Right-to-left check: Hebrew mixed with English and digits must not reorder ("jump") while words are revealed
- [ ] Real-world test: 60 min conversation, screen on; latency, heat, battery (Gate A′ in the app)
- [ ] Show it to 2–3 deaf testers → **Gate C**

### Phase 4: Scale-up training

- [ ] More ivrit.ai data + pseudo-labeled audio (teacher: ivrit.ai large-v3-turbo)
  - [ ] **Filter teacher labels before training on them**, or Whisper's hallucinations become training targets:
    - agreement between teacher and current Nemotron (CER below a threshold)
    - repetition filter
    - script/language filter
    - no text where VAD finds no speech
  - [ ] Check which ivrit.ai datasets are human-transcribed and which are machine-transcribed (machine labels carry the same risk)
- [ ] More code-switching and far-field data, based on prototype errors
- [ ] Quantization-aware training (reliable 8-bit, NPU)
- [ ] *Only if the cheap levers (q4, threads, chunk size, VAD, newer phones) aren't enough:* distill into a smaller streaming student (~100–150M). This is a research project of its own: new architecture, probably a new tokenizer, thousands of hours of teacher-labeled audio. Start from an existing small streaming FastConformer checkpoint if a suitable one exists (unverified), not from scratch.

### Phase 5: Later

- [ ] Choose the chip and convert to its NPU toolchain (Qualcomm QNN / AI Hub, MediaTek NeuroPilot, or LiteRT); get a newer Snapdragon phone
- [ ] Smart glasses: phone transcribes; glasses display text (Bluetooth LE). Check right-to-left Hebrew rendering when choosing glasses: many SDKs handle it badly.

*(The former "optional quick phone tests" moved into Phase 1 → Phone re-tests.)*

### Plan B track (only if a trigger happens)

- [ ] Fine-tune Whisper `base` (standard vocabulary) on ivrit.ai data → GGML `q8_0` → phone
- [ ] WhisperEngine: rolling buffer + commit logic (WhisperLive-style), VAD mandatory

---

## 8. User research (Phase 1, not a side track)

It decides several open decisions and Gate C, so it starts now.

- [ ] **Define the primary user.** Deaf people whose first language is Israeli Sign Language, or hard-of-hearing / late-deafened people (often older)? Reading fluency, phones and needs differ.
- [x] ~~Contact the Association of the Deaf in Israel and Shema~~ skipped (D-018)
- [ ] Short survey:
  - phone model, OS, age range
  - where and when captions are needed
  - ~~is offline/privacy a must?~~ decided: yes (D-017)
- [ ] Ask about features that may matter more than WER:
  - who is speaking
  - a vibration when someone starts talking to them, or says their name
  - font size and contrast
- [ ] Ask 2–3 deaf people early: English words in Latin or Hebrew letters? (needed before Phase 2; no prototype needed)
- [ ] Recruit 2–3 early testers; consent form for any recording
- [ ] Benchmark Google Live Transcribe in Hebrew **in Phase 1** (an afternoon; it's the bar to beat and helps set the Gate B targets). The leaderboard's Google entry is the cloud API, not necessarily Live Transcribe.
- [ ] Define the Gate C measures with the testers

---

## 9. Open decisions

Kept in one place only: `docs/DECISIONS.md` → **Open decisions**.

---

## 10. Known risks

1. **Thin speed headroom on mid-range phones** (hot RTF ~0.6–0.7, measured on English, screen off). Mitigation: Gate A′; q4, 1120 ms chunks, threads, VAD pauses, NPU, newer phones, distillation last.
2. **Hebrew quality after fine-tuning unknown.** Mitigation: Gate B on several sets, two data points; scale data up to a set limit.
3. **Fine-tuning could bring back hallucinations**, and teacher labels can bring in Whisper's. Mitigation: a larger hallucination suite after every training run; filter teacher labels.
4. **16 GB GPU is tight for 600M RNNT training** (optimizer state alone ~9.6 GB). Mitigation: fused joint/loss, bf16, small batches, shorter samples, activation checkpointing, 8-bit AdamW; smoke run first; rent a GPU for big runs.
5. **Code-switching:** one language prompt per stream, and no public code-switched dataset. Mitigation: natural code-switching in ivrit.ai podcasts; own data.
6. **Distant speech produces fluent wrong text** (seen with Whisper). Mitigation: far-field test clips and data; confidence check (Phase 2) and confidence-based fading.
7. **Licenses:** confirm OpenMDW-1.1 and each dataset before training (rule 6) and shipping.
8. **Data leakage:** exclude the whole eval-d1 podcast series; the guard needs `--source-col`; validation split by episode.
9. **Storage:** 500 GB is not enough.
10. **Hebrew may decode slower than English:** more tokens per second → more RNNT decoder steps. Mitigation: tokenizer check, Hebrew heat test (Phase 1).
11. **Screen-on battery and heat** were never measured; the app keeps the screen on. Mitigation: Gate A′.
12. **Android may kill long sessions:** background mic limits, battery optimization. Mitigation: microphone foreground service + exemption, tested in the minimal app.
13. **Young upstream runtime:** NeMo-Speech.cpp v0.1.0, and it's locally patched. Mitigation: pin the commit; sherpa-onnx as fallback (D-014).
14. **Right-to-left rendering:** mixed Hebrew/English/digits can reorder on phones and glasses. Mitigation: RTL check in Phase 3 and when choosing glasses.
15. **Unvalidated premise:** "fully on-device" hasn't been checked with users. Mitigation: survey question (§8); open decision.
16. **No time limits:** solo project, no deadlines, so the triggers can't fire. Mitigation: time budget as an open decision.

---

## 11. Key resources

| Resource | Use |
|---|---|
| huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b | Plan A model |
| github.com/NVIDIA/NeMo-Speech.cpp | Phone runtime, GGUF converter (`convert_model.py`), native SDK |
| Hugging Face blog: "How to Fine-Tune Nemotron 3.5 ASR…" | Fine-tuning guide |
| github.com/NVIDIA/NeMo | Training toolkit |
| huggingface.co/spaces/ivrit-ai/hebrew-transcription-leaderboard | Baselines, test sets, evaluation code |
| huggingface.co/ivrit-ai | Hebrew datasets; teacher model (whisper-large-v3-turbo) |
| openmdw.ai/license/1-1 | Model license |
| github.com/ivrit-ai/asr-training (`evaluate_model.py`) | Leaderboard evaluation + normalization |
| github.com/k2-fsa/sherpa-onnx | Fallback runtime (Android SDK) for trigger 2 |
| github.com/ggml-org/whisper.cpp | Plan B runtime |
| github.com/collabora/WhisperLive | Plan B commit logic reference |
| Silero VAD | Voice activity detection |
| Termux (F-Droid) | Phone tests |
