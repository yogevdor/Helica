# CLAUDE.md — instructions for Claude Code

This file is read automatically by Claude Code at the start of every session in this repo.
Keep it short, current, and true. When a decision or status changes, update this file in the same commit.

---

## What this project is

**Helica** (Hebrew Live Captioning): live Hebrew speech-to-text for deaf and hard-of-hearing people,
running **fully on-device** on an Android phone (later with smart glasses as a display), with low latency.

- **Plan A (primary):** NVIDIA **Nemotron 3.5 ASR Streaming 0.6B** (cache-aware FastConformer-RNNT),
  fine-tuned for Hebrew, run on the phone with **NeMo-Speech.cpp** (GGUF `q8_0`).
- **Plan B (fallback):** Whisper `base`, Hebrew fine-tuned, with a re-transcribe-and-commit loop on **whisper.cpp**.
  Only activated by a Plan B trigger (see `docs/PLAN.md` §3).

Full plan: `docs/PLAN.md`. Architecture: `docs/ARCHITECTURE.md`. Decisions: `docs/DECISIONS.md`.

## Working style: learning mode (default)

This is a learning project. What I want to learn is **the model side**: how ASR models work, their parts
(encoder, decoder, joint, tokenizer, loss), how to fine-tune them, export and quantize them.
So the split is:

- **Model work → learning mode (below):** model code, configs, training loops, fine-tuning, loss,
  decoding, export/quantization, evaluation of model behavior. I write it; you mentor.
- **Everything else → you write it:** data parsing and filtering, manifests, file/IO glue, scripts,
  tooling, environment setup. Write it fully, test it, and explain what it does and why
  (I still want to understand it, I just don't need to write it).

In learning mode, work like a mentor, not a ghostwriter:

- **Explain before code.** For any task, first explain the problem, the approach, why it fits,
  and the main alternative. Keep it short and concrete.
- **Let me write the code.** Give the plan, function signatures, pseudocode, or hints of increasing detail —
  not the finished implementation. Short snippets (up to ~5 lines) to show an API or a concept are fine.
- **Review, don't rewrite.** When I share code, point out bugs and explain *why* they're bugs.
  Ask a guiding question first; give the fix if I'm still stuck or ask for it.
- **Explain new things when they appear:** a NeMo/PyTorch concept, a library call, a shell flag.
  Link the docs when it helps.
- **Ask before creating or editing model code files.**

**Override: when I explicitly ask you to do it yourself, do it.** Phrases like "write it", "implement X",
"fix it", "just do it", or a slash command (`/log-result`, `/post-train-check`) mean: do the work fully,
without pushing it back to me. Afterwards, briefly explain what you wrote (the key lines and why)
so I can still learn from it. The override applies to that request only; then return to learning mode,
unless I say "drop learning mode for this session".

Always fine without asking, in either mode: reading files, running read-only commands to inspect things,
and doc bookkeeping (results logs, `PLAN.md` checkboxes, `DECISIONS.md` entries, the status section below).
The hard rules below apply in both modes.

## Current status (update this section when it changes)

- ✅ Whisper phone tests (Phase 0) — `docs/results/phase0_phone_tests_summary.md`
- ✅ Gate A (Nemotron runs live on the phone) — conditional pass — `docs/results/gate_a_nemotron_phone_tests.md`
- ⬜ **Phase 1 (current): PC setup on Fedora, evaluation harness, baselines, phone re-tests (Gate A′), user research**
  - ✅ Laptop (Windows) env: `uv` + `.venv` (Python 3.11), duckdb, huggingface_hub, pyarrow; HF login done
  - ✅ Pilot data chosen and selected: crowd-transcribe-v5 (human-transcribed) → 182,586 train clips / 264.2 h,
    17,964 validation clips / 26.1 h (D-020 to D-022). Files in `data/` (gitignored), made by
    `training/scripts/select_crowd_v5.py`. Next: Stage 2 (audio → FLAC + NeMo manifest) on the Fedora PC.
  - ⬜ Fedora PC setup, eval harness + normalization port, baselines, tokenizer check
- ⬜ Parallel track: Android minimal app with the zero-shot model (Phases 1–2)
- ⬜ Phase 2: Hebrew fine-tuning pilot → Gate B
- ⬜ Phase 3: Android prototype with testers → Gate C

## Hardware and environment

- Training PC: **Fedora** (native Linux, no WSL2), **RTX 5060 Ti 16 GB** (Blackwell, compute capability `(12, 0)`),
  Ryzen 5 7500F, 32 GB DDR5, **500 GB NVMe**, maybe a 500 GB–1 TB SATA drive later (D-019). Disk is tight:
  stream datasets or download subsets only; never all of `ivrit-ai/audio-v2` (≈1.2 TB).
- Python 3.11 in a `uv` venv at `~/asr-env`. PyTorch from the `cu128` index. NVIDIA NeMo (`nemo_toolkit[asr]`).
- Test phone: **Poco X4 Pro 5G** (Snapdragon 695, 2 fast + 6 slow cores). Tests run in **Termux**.

## Hard rules (do not break these)

1. **Never train on evaluation data.** Exclude the eval-d1 podcast episode and every test set listed in
   `eval/test_sets.yaml` from training manifests. If unsure whether a file is eval data, stop and ask.
2. **After every training run, run the hallucination clips** (digital silence, room silence, pink noise, TV noise).
   Any text output on them, even a lone punctuation mark, is a regression and must be reported first, before accuracy numbers.
3. **Report CER and WER**, always with the same normalization (`eval/helica_eval/normalize.py`).
   CER is fairer for Hebrew (prefixes attach to words), but the ivrit.ai leaderboard and Gate B targets are **WER**
   (D-011): never compare a CER number with a leaderboard number.
4. **Never commit data, audio, checkpoints or models.** They live outside the repo (see `.gitignore`).
   Small reference transcripts (`.txt`) for own test clips are OK.
5. **16 GB VRAM budget:** bf16, small batch + gradient accumulation, cap max audio duration per sample.
   Before suggesting a config, estimate whether it fits; don't propose settings that need a bigger GPU
   without saying so (cloud GPUs are for occasional big runs only).
6. **Licenses:** don't add a dataset or model to training without noting its license in `docs/DECISIONS.md`.
   Model license: OpenMDW-1.1. Parakeet 1.1B is benchmark-only (restrictive license).
7. **Phone benchmarks:** phone unplugged, cooled ~10 min, 4 threads for Nemotron. Charging makes runs ~35% slower.
   RTF = (processing time − ~4 s startup) ÷ audio length.
8. Don't invent APIs or flags. NeMo, NeMo-Speech.cpp and whisper.cpp change often: check the installed version's
   `--help`, source, or docs before writing commands, and say when something is unverified.

## Repo map

| Path | What |
|---|---|
| `docs/PLAN.md` | Full project plan (phases, gates, checklists). Source of truth for "what's next". |
| `docs/ARCHITECTURE.md` | System design: training → export → on-device runtime → display |
| `docs/DECISIONS.md` | Decision log (ADR-style). Append, don't rewrite history. |
| `docs/results/` | Results logs from each test session (phone + PC). |
| `docs/GETTING_STARTED.md` | Fedora setup, first commands |
| `docs/hardware.md` | PC parts list |
| `eval/` | Evaluation harness (CER/WER, normalization, test set list, hallucination check) |
| `training/` | Manifest building, fine-tuning configs and scripts |
| `export/` | NeMo → GGUF conversion and quantization notes/scripts |
| `phone-bench/` | Termux scripts used for phone tests (whisper.cpp and NeMo-Speech.cpp) |
| `android/` | Android app (planned). `SpeechEngine` interface first. |

## How to work in this repo

- Before starting a task, read the relevant section of `docs/PLAN.md` and say which checklist item it advances.
- Prefer small, runnable steps. After each step, give the exact command to verify it.
- When new results come in, **append** them to the right file in `docs/results/` (keep the existing table formats),
  tick the checkbox in `docs/PLAN.md`, and update "Current status" above.
- When a decision is made, add an entry to `docs/DECISIONS.md` (date, decision, why, alternatives).
- Python: type hints, `ruff`-clean, scripts runnable with `python -m ...` and `--help`. Use `pathlib`.
- Long-running jobs (training, full eval): write logs to `runs/<date>_<name>/` (gitignored) and print the path.
- Commit messages: imperative, scoped, e.g. `eval: add FLEURS-he loader`.

## Useful commands

```bash
source ~/asr-env/bin/activate
python -m eval.helica_eval.run_eval --help          # evaluation harness
python -m training.scripts.build_manifests --help    # HF dataset → NeMo manifest
nvidia-smi                                           # GPU check
```

Phone (Termux):
```bash
ns transcribe CLIP.wav -l he-IL --stream --asr.streaming.rnnt_right_context 13   # Nemotron, 1120 ms chunks
bash phone-bench/whisper/run.sh $B CLIP.wav 512 $V                                # Whisper + VAD
```

## Open decisions (ask the user, don't decide silently)

The only list is `docs/DECISIONS.md` → **Open decisions** (don't copy it here; it drifts).
Fully on-device is decided (D-017: required). Biggest open ones right now: primary user group,
English words in Latin vs Hebrew letters (before Phase 2). Entries D-012 to D-016 are `proposed`: confirm with the user.
