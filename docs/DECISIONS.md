# Decision Log

Append new decisions at the bottom. Don't rewrite old entries; if a decision is reversed, add a new entry that says so and link back.

Format: **ID — Title** · date · status (accepted / superseded / proposed)

---

### D-001 — Android only · Sept 2026 · accepted
- **Decision:** Target Android. Apple dropped.
- **Why:** Open NPU toolchains, Termux for fast experiments, cheaper test devices.

### D-002 — Prototype before training · Sept 2026 · accepted
- **Decision:** Measure on the real phone first, then train.
- **Why:** Phone results define what training must fix (window size, latency budget, error types).

### D-003 — Test phone: Poco X4 Pro 5G (Snapdragon 695) · Sept 2026 · accepted
- **Why:** Represents a typical mid-range user phone. NPU is weak → NPU work later needs a newer phone.

### D-004 — Train on own PC (RTX 5060 Ti 16 GB), Fedora native · Sept–Oct 2026 · accepted
- **Why:** Free iteration. Cloud GPUs only for occasional big runs. Native Linux (no WSL2) for driver/CUDA simplicity.

### D-005 — Silero VAD mandatory for Whisper · Sept 26, 2026 · accepted
- **Why:** Stock Whisper `base` invented text on all 4 non-speech clips; VAD removed it fully while keeping speech intact.

### D-006 — Nemotron 3.5 ASR Streaming 0.6B is Plan A; Whisper `base` is Plan B · Oct 1, 2026 · accepted
- **Decision:** Main engine = Nemotron (cache-aware FastConformer-RNNT) via NeMo-Speech.cpp, GGUF `q8_0`.
- **Why:** Gate A conditional pass (RTF 0.43 cool, ~0.59 avg / 0.68 worst hot); true streaming (no re-transcription); no invented text on silence/noise even without VAD; Hebrew in tokenizer; outputs punctuation.
- **Alternatives:** Whisper `base` + commit loop (works, but hallucinates without VAD, re-transcribes, needs own Hebrew fine-tune); wav2vec2/Omnilingual CTC (no punctuation, weaker LM).
- **Reversal triggers:** see `PLAN.md` §3 "Plan B triggers".

### D-007 — Phone runtime defaults: 1120 ms chunks, 4 threads, model loaded once · Oct 1, 2026 · accepted
- **Why:** 1120 ms was lightest; 6/8 threads were slower (threads land on slow cores). 560 ms remains an option for testers.

### D-008 — Evaluate with the ivrit.ai leaderboard sets and code · Oct 2026 · accepted
- **Why:** Numbers directly comparable with published Hebrew models; no harness from scratch.

### D-009 — Whisper `base` Hebrew fine-tune dropped for now · Oct 2026 · accepted
- **Why:** Plan A chosen. Revisit only if a Plan B trigger happens.

### D-010 — Public GitHub repo "Helica" · Oct 26, 2026 · accepted
- **Decision:** Code, docs and results in a public repo. Data, audio and model weights never committed.

### D-011 — Leaderboard comparisons use WER; CER always reported next to it · Oct 26, 2026 · accepted (records a fact)
- **Decision:** Every number compared with the ivrit.ai leaderboard is **WER**, after the leaderboard's normalization (`evaluate_model.py` in `ivrit-ai/asr-training`). CER is always reported alongside, because it's fairer for Hebrew (prefixes attach to words).
- **Why:** The leaderboard states it scores WER with jiwer. Earlier docs said "error rate", so the Gate B target (0.171 → ≤ 0.08) had no unit.

### D-012 — Gate A target relaxed after the fact; re-test as Gate A′ · Oct 26, 2026 · proposed
- **Decision:** Record that Gate A's "RTF ≤ 0.5 when hot" target was missed (hot plateau 0.59 avg / 0.68 worst) and passed as "conditional" after the measurement. Add Gate A′ (PLAN §3): Hebrew audio, screen on, 60 min, threads 2/3/4, q8 vs q4, peak RAM.
- **Why:** The Gate A run was easier than real use:
  - English audio (`jfk.wav` × 6, `-l en-US`)
  - screen off between runs
  - 14 minutes only
  - the "VAD will pause processing" margin assumed, not measured; in group conversations or with a TV on, VAD will report speech most of the time
- **Alternatives:** Keep Gate A as passed with no re-test. Rejected: Phase 3 would be the first time real conditions are measured.

### D-013 — Silero VAD for Nemotron (battery), with audio kept after speech ends · Oct 26, 2026 · proposed
- **Decision:** Use Silero VAD in front of Nemotron to pause inference during silence. Keep feeding ~0.5–1 s of audio after speech ends before pausing, and ~300 ms before speech starts.
- **Why:** Nemotron didn't hallucinate without VAD, so VAD is for battery, not correctness. Cutting silence out completely joins utterances together in the model's cache, and end-of-sentence punctuation may depend on pauses. Compare punctuation and WER with and without VAD.
- **Alternatives:** No VAD (simplest; costs battery during silence).

### D-014 — Phone runtime: NeMo-Speech.cpp; sherpa-onnx as fallback · Oct 26, 2026 · proposed
- **Decision:** NeMo-Speech.cpp (GGUF) stays the runtime. Pin its commit hash in builds and results. If Android integration fails (trigger 2), try **sherpa-onnx** with the same model before switching to Whisper.
- **Why:** NeMo-Speech.cpp worked in Termux (Gate A), but it's young (v0.1.0) and locally patched (`NS_THREADS`). sherpa-onnx has an Android SDK and already runs Nemotron 3.5 ASR Streaming on desktop (INT8, chunks 80 ms–1.12 s). Unverified on Android with a fine-tuned model.
- **Alternatives:** ONNX Runtime / ExecuTorch directly (more glue code); Whisper (Plan B, weaker model).

### D-015 — q8_0 for Nemotron is provisional · Oct 26, 2026 · proposed
- **Decision:** Keep q8_0 as the default until q4_0 is tested on the phone (speed + output text).
- **Why:** q8_0 was chosen because 8-bit beat 5-bit *for Whisper* on the Snapdragon 695. ggml has fast q4_0 paths on ARM with dotprod; Nemotron was never tested at 4 bits. Accuracy loss at 4 bits must be measured, not assumed.

### D-016 — Plan B (Whisper `base`) scope narrowed · Oct 26, 2026 · proposed
- **Decision:** Each Plan B trigger gets its own fallback (PLAN §3). Whisper `base` is the fallback for trigger 4 (license), and for trigger 2 only if sherpa-onnx also fails. It's not the answer to trigger 1 (accuracy) or trigger 3 (heat/battery).
- **Why:**
  - **Accuracy:** a 74M Whisper won't beat a fine-tuned 0.6B Nemotron (small Hebrew Whispers are ~0.38 WER).
  - **Heat/battery:** Whisper `base` re-transcribes ~7 s every ~1.2 s and used about the same battery (15–20%/h) in Phase 0.
- **Alternatives:** Keep Whisper as the fallback for all triggers (earlier plan).

### D-017 — Fully on-device is a hard requirement · Nov 11, 2026 · accepted
- **Decision:** Captions must work offline. No cloud model, not even as an optional online mode.
- **Why:** User requirement (stated directly; no survey).
- **Consequence:** "Hybrid" is off the table. Accuracy gains must come from training, model choice and on-device optimization.

### D-018 — No contact with deaf associations for now; time budget not tracked · Nov 11, 2026 · accepted
- **Decision:** Skip contacting the Association of the Deaf in Israel and Shema. Don't track a time budget.
- **Consequence:** Open questions that were to be settled by user research (primary user group, English words in Latin vs Hebrew letters) are decided by the user directly. Plan B triggers that depend on time limits have no deadline.

### D-019 — Storage: 500 GB NVMe, plus an optional 500 GB–1 TB SATA drive · Nov 11, 2026 · accepted
- **Decision:** No 2 TB drive. Datasets that are being trained on live on the NVMe; the SATA drive (if added) holds the Hugging Face cache and archives.
- **Why:** Hardware available.
- **Consequence:** Never download all of `ivrit-ai/audio-v2` (≈1.2 TB): stream it, or download only the subset in use. A 200 h pilot is ~15 GB as 16 kHz FLAC (plus the downloaded source). Phase 4 (thousands of hours) needs the SATA drive and careful pruning; if it is a hard disk (not SSD), reading many small audio files from it may be too slow to train from directly; measure before relying on it.

### D-020 — Pilot trains on human-transcribed data first · Nov 15, 2026 · accepted
- **Decision:** The Phase 2 pilot uses ivrit.ai's human-transcribed sets: `crowd-transcribe-v5` (~300 h, volunteers correcting automatic transcripts) and possibly `crowd-recital-whisper-training` (~50 h, read Wikipedia snippets). The machine-labeled `audio-v2` + `audio-v2-transcripts` (>20k h) comes later, for scale-up.
- **Why:** Cleaner labels, so the pilot measures whether fine-tuning works rather than label noise. The crowd sets are already short segments (avg ~5 s), so no episode slicer is needed for the pilot.
- **Watch out:** crowd-transcribe segments were sampled from the ivrit.ai corpus, which eval-d1's episode also comes from. Check the eval-d1 source (Mindset, 2018.10.07) against these sets too, not only audio-v2.
  - **Checked Nov 16, 2026 (DuckDB on the `uuid` column):** `Mindset/2018.10.07%` → **0 rows in train**, **74 rows in test** (203,827 train rows total). No training leak. If crowd-transcribe `test` is used as the validation set, drop those 74 rows so checkpoint selection isn't tuned on eval-d1 audio. audio-v2 still unchecked.
- **Alternatives:** Start with machine-labeled audio-v2 (more hours, needs the slicer and label filters).

### D-021 — crowd-transcribe-v5: duplicate and empty-text rules · Nov 16, 2026 · accepted
- **Facts (checked with DuckDB on a local copy of all non-audio columns, `data/crowd_v5_meta.parquet`):**
  - train 203,827 rows / 186,937 clips / 16,639 episodes; test 20,711 rows / 18,554 clips / 441 episodes.
  - **0 episodes shared between train and test**: ivrit.ai split by episode, so `test` is a clean validation set (minus the 74 eval-d1 rows, D-020).
  - 17,030 clips have two transcripts (two volunteers, usually one marked `is_retranscribe`). 6,201 differ in any way; **4,856 differ in the words** (ignoring punctuation and spaces).
  - Empty `sentence`: 84 train, 8 test.
- **Decision:**
  - Train on `sentence` (the corrected text), never `orig_sentence`.
  - Clip with two transcripts: same words → keep one copy; different words → drop the clip (~2% of clips).
  - Drop rows with empty `sentence`.
- **Why:** Two volunteers disagreeing means the label is uncertain; dropping ~2% of clips is cheap for cleaner pilot labels.
- **Alternatives:** keep both copies (likely ivrit.ai's practice; double weight, conflicting labels); keep one at random (keeps the disagreement noise).

### D-022 — Pilot filter list for crowd-transcribe-v5 · Nov 16, 2026 · accepted
- **Facts:** train 295.8 h, test 30.3 h (before filters, duplicates included). `skipped` = 0 (every row human-checked). `unintelligible`: 57 train / 1 test. `foreign_language` 9, `multiple_speakers` 3, `noisy` 1 (volunteers rarely used these flags). Clips are 2.05–29 s; train hours in clips > 20 s: 37.7, > 25 s: 27.0.
- **Rules:**
  1. Text = `sentence`, trimmed (never `orig_sentence`).
  2. Drop empty `sentence`, `unintelligible`, `foreign_language`.
  3. Duplicates per D-021 (same words → keep one; different words → drop the clip).
  4. Validation = crowd-transcribe `test` with the same rules, minus the 74 eval-d1 rows (D-020).
  5. Max length: **30 s** (keep everything) if the 200-step memory test passes with duration-based batching; otherwise 20 s (loses ~38 h of train).
  6. No minimum length needed; keep `multiple_speakers` and `noisy`.
- **Result (`select_crowd_v5.py`, max 30 s, Nov 16, 2026):** train **182,586 clips / 264.2 h** (4,215 clips dropped for disagreeing transcripts, 0 too long); validation (test) **17,964 clips / 26.1 h** (522 disagree, 62 eval-d1 clips excluded). Train contains no Mindset clips at all.
- **Why:** Uses the volunteers' own quality signals; the only real data loss is the length limit, decided by measured GPU memory, not guessed.

---

## Licenses register

Fill in before any dataset/model is used for training or shipped. **Blocks Phase 2** (CLAUDE.md rule 6): nothing goes into training while its row says "?".

| Item | License | Allows fine-tuning? | Allows shipping in app? | Checked on |
|---|---|---|---|---|
| nvidia/nemotron-3.5-asr-streaming-0.6b | OpenMDW-1.1 | ? | ? | |
| ivrit.ai training datasets (list each, and which version applies: v1 = audio-base/-vad/-transcripts, v2 = crowd datasets) | ivrit.ai License v1/v2 = CC BY 4.0 **plus** a purpose limit: use only for training AI models (v2 also: academic research). No simulating speakers' voice or likeness. Credit ivrit.ai. | ✅ (the permitted purpose) | **Not stated** in the license text: it says nothing about trained models. Check the License FAQ; models trained on it are published openly in practice (ivrit.ai's own, zipformer2 CC-BY-4.0, kol-stream Apache-2.0) | Nov 11, 2026 (license page; FAQ not yet read) |
| ivrit-ai/audio-v2 (>20k h, 1.21 TB, gated: access granted Nov 11, 2026) | ivrit.ai License **v1** (per its card): training AI models only | ✅ | See row above (not stated) | Nov 11, 2026 |
| ivrit-ai/eval-d1, eval-whatsapp (eval only) | ivrit.ai License (gated; confirm version on each card) | — (never train) | — | Nov 11, 2026 (see row above). Don't commit their audio or reference transcripts (e.g. per-clip CSVs) to the public repo: aggregate numbers only |
| SASpeech, FLEURS-he, Common Voice 17 he, hebrew_speech_kan (eval only) | ? | — (never train) | — | |
| English training data (choose one) | ? | ? | — | |
| Noise corpus for the hallucination suite (choose one) | ? | — | — | |
| ivrit-ai/whisper-large-v3-turbo (teacher) | ? | — | — (teacher only) | |
| Silero VAD | ? | — | ? | |
| NeMo-Speech.cpp | ? | — | ? | |
| sherpa-onnx (fallback runtime) | ? | — | ? | |
| Parakeet RNNT 1.1B | NVIDIA Community License | Benchmark only | ❌ | |

## Open decisions

The only list of open decisions; PLAN.md and CLAUDE.md link here. Ask the user, don't decide silently.

| Question | Options | How to decide |
|---|---|---|
| ~~**Fully on-device: hard requirement or preference?**~~ → **decided: D-017 (on-device only)** | On-device only / hybrid (cloud model when online, on-device offline) | User research (§8 survey: is offline/privacy a must?). Cloud models are ~3× more accurate on eval-d1 today. |
| **Primary user group** | Deaf, Israeli Sign Language first / hard-of-hearing or late-deafened (often older) / both | The user decides (no association contact, D-018). Affects display, latency vs accuracy, phones to support. |
| ~~**Time budget**~~ → **dropped: not tracked (D-018)** | Hours per week, target date per phase | You. Needed so the Gate B "scale-up limit" and Plan B triggers can actually fire. |
| Gate B targets beyond eval-d1 | eval-whatsapp WER ≤ ?, real-world set WER ≤ ? | Live Transcribe benchmark + user research (Phase 1) |
| Chunk size | 1120 ms (default) / 560 ms | Testers |
| English words in captions | Latin vs Hebrew letters | Text-convention audit of ivrit.ai data + quick question to 2–3 deaf people. **Before Phase 2**: can't be changed after training. |
| Punctuation and number style in training text | Follow ivrit.ai transcripts (default) / re-normalize | Text-convention audit, **before Phase 2**. Numbers can be converted at display time. |
| Caption distant/background speech | Show / dim / hide | Testers |
| Distilled smaller student | Yes / No | Only if Gate A′ levers (q4, threads, chunk size, VAD, newer phones) aren't enough |
| Repo license | MIT / Apache-2.0 / other | Before first public push |
