# Evaluation harness

Every model (zero-shot, fine-tuned, exported, quantized) goes through the same three checks:

1. **CER / WER** on the leaderboard sets + own clips + real-world set, at the chunk sizes used on the phone.
   Leaderboard comparisons are **WER** (D-011).
2. **Hallucination suite**: non-speech clips must produce no output at all, checked on the raw text (even "." fails). Hard gate.
   `summary.json` also reports `n_empty_ref_with_text`: utterances with an empty reference where the model produced text.
3. **Per-clip log** (`per_clip.csv`) to study errors, not just averages.

```bash
python -m eval.helica_eval.metrics                                    # self-test
python -m eval.helica_eval.run_eval score --ref ref.jsonl --hyp hyps.jsonl --name nemotron-zs@1120_eval-d1
```

| File | Purpose |
|---|---|
| `helica_eval/normalize.py` | Hebrew normalization (provisional — align with the ivrit.ai leaderboard code) |
| `helica_eval/metrics.py` | Corpus CER/WER via jiwer |
| `helica_eval/hallucination.py` | Non-speech gate |
| `helica_eval/run_eval.py` | CLI: score hypotheses; engine transcription TODO |
| `test_sets.yaml` | Test sets, zero-shot baselines, training exclusion list |
| `own_clips/` | Reference transcripts (`.txt`) for own recordings. Audio is NOT committed. |

## Phase 1 checklist

- [ ] Align normalization with `evaluate_model.py` in `github.com/ivrit-ai/asr-training` (what the leaderboard links to)
- [ ] Reproduce Nemotron zero-shot eval-d1 ≈ 0.171 WER (proves the harness)
- [ ] Implement `nemo` engine with cache-aware streaming at 560 / 1120 ms
- [ ] Implement `whisper-ct2` engine for the ivrit.ai teacher baseline
- [ ] Write reference text for every own clip
- [ ] Real-world set (30–60 min, hand-transcribed) and a hallucination suite grown to 30–60 min (see `test_sets.yaml`)
