---
description: Full checklist after a training run or export (hallucination gate, CER/WER, parity)
---

Run directory / model: $ARGUMENTS

Go through these in order and stop at the first failure:

1. **Hallucination suite** (`eval/helica_eval/hallucination.py`) on every non-speech clip in `eval/test_sets.yaml`. Any output, even a lone punctuation mark → FAIL, report it first. Also report `n_empty_ref_with_text` from the eval summaries.
2. **WER and CER** on every set in `eval/test_sets.yaml` at 560 and 1120 ms chunks, via `eval.helica_eval.run_eval`. Compare **WER** with the zero-shot baselines in the yaml and the ivrit.ai models in `docs/PLAN.md` §6.2 (the leaderboard is WER, D-011). Gate B looks at eval-d1, eval-whatsapp and the real-world set together.
3. **Own clips + real-world set**: clean, noisy, 2.5 m, code-switching. Show ref vs hyp side by side for the worst 5 clips.
4. If a GGUF export exists: **parity check** on ~200 eval utterances (PyTorch vs GGUF q8_0, target CER gap < 0.5 points, plus exact-match rate).
5. Write everything to `docs/results/` with `/log-result`, and say whether this moves Gate B.
