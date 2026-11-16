# Training

```
pilot:     crowd-transcribe-v5 ──select_crowd_v5.py──▶ clip list ──Stage 2 (TODO)──▶ NeMo manifest + 16 kHz FLAC
scale-up:  HF dataset ──build_manifests.py (draft)──▶ NeMo manifest + 16 kHz FLAC
                     ▲ both read eval/test_sets.yaml (exclusions)
then:      manifest ──NeMo fine-tune──▶ .nemo ──▶ eval/ ──▶ export/
```

| File | Status |
|---|---|
| `scripts/select_crowd_v5.py` | **Pilot** (D-020 to D-022): filters crowd-transcribe-v5 metadata, applies the leakage guard, writes the selected-clip lists |
| `scripts/build_manifests.py` | **Scale-up draft, not used for the pilot:** generic HF (audio + text rows) → manifest, resampling, duration filter, leakage guard (prefix match, needs `--source-col`), shuffle, per-source cap, episode-held-out validation. Untested on a real dataset; `audio-v2` keeps transcripts in a separate dataset, so it needs changes first |
| `configs/pilot_16gb.yaml` | Planning values + VRAM estimate for the 16 GB pilot; turn into a real NeMo config from NVIDIA's guide |

## Pilot (Phase 2)

1. Decide text conventions (punctuation, English words, numbers) — `docs/DECISIONS.md`.
2. `exclude_sources` in `eval/test_sets.yaml` is filled (eval-d1 episode, prefix match; see the note there on the whole series).
3. Select clips with `select_crowd_v5.py` (done, see below), then Stage 2: audio → FLAC + NeMo manifests (TODO).
   Validation = crowd-transcribe `test` split. Still to decide: 10–20 % English/code-switched data and a 50 h subset for the second data point.
   (`build_manifests.py` is for later datasets; see the table above.)
4. 200-step smoke run (peak VRAM, steps/s), then fine-tune with bf16, fused joint/loss, small batch + grad accumulation, max 20 s samples.
5. Run the full evaluation (hallucination suite first, then WER/CER at 560/1120 ms) → `docs/results/`.
6. Decide Gate B.

Logs and checkpoints: `runs/<date>_<name>/` (gitignored). Data: `$HELICA_DATA` (see D-019 for storage).

## Pilot data: crowd-transcribe-v5 (human-transcribed, D-020 to D-022)

```
HF parquet (text columns only, via DuckDB) ─▶ data/crowd_v5_meta.parquet           (done once, ~16 min)
data/crowd_v5_meta.parquet ─select_crowd_v5.py─▶ data/crowd_v5_{train,test}_selected.jsonl   (seconds)
selected.jsonl + audio (Stage 2, Fedora PC) ─▶ 16 kHz FLAC + NeMo manifest                     (TODO)
```

`python -m training.scripts.select_crowd_v5 [--max-sec 20]` applies the D-022 rules and prints what each
rule removed. Everything under `data/` is gitignored (ivrit.ai license: don't republish transcripts).
