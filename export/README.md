# Export: NeMo → GGUF → q8_0 → phone

```
.nemo ──convert_model.py (NeMo-Speech.cpp)──▶ GGUF f16 ──quantize──▶ GGUF q8_0
                                                                  │
                     parity check on PC CPU (same clips: PyTorch vs GGUF) ◀──┘
                                                                  │
                     phone check in Termux (same clips, text + RTF) ◀──┘
```

## Steps (verify exact flags against the NeMo-Speech.cpp version you build)

1. Convert the fine-tuned checkpoint with NeMo-Speech.cpp's `convert_model.py` → GGUF.
2. Quantize to `q8_0` (provisional default, D-015: 8-bit was faster than 5-bit on the Snapdragon 695 for Whisper). Also try `q4_0` for Nemotron and compare speed + text.
3. **Parity check on the PC:** run ~200 eval utterances (the own clips alone are too few to mean anything) through NeMo PyTorch and through NeMo-Speech.cpp (CPU build on Fedora), same chunk size, streaming mode on both sides. Score both with `eval/`. Target: CER gap < 0.5 points absolute; also report the exact-match rate of the two outputs.
4. **Hallucination suite on the GGUF model** — quantization can change silence behavior.
5. Copy to the phone, run `phone-bench/nemotron/` tests: same text as the PC GGUF output, RTF in line with Gate A.

## Later: NPU (Phase 5)

Encoder only, per-chunk, static shapes, caches as explicit inputs/outputs; prediction net + joint + greedy decode stay on CPU. See `docs/ARCHITECTURE.md` §2.

TODO: `parity_check.py` (PyTorch vs GGUF on the same manifest, writes both hyps.jsonl files and calls `eval.helica_eval.run_eval score`).
