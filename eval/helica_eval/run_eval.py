"""Evaluation entry point.

Two modes:

1. Score existing hypotheses (works today, engine-independent):
       python -m eval.helica_eval.run_eval score --ref manifest.jsonl --hyp hyps.jsonl --name nemotron-zs@1120
   ref: NeMo manifest lines with "audio_filepath" and "text".
   hyp: JSON lines with "audio_filepath" and "pred_text" (NeMo's transcribe scripts write this format).

2. Transcribe + score with an engine (TODO, Phase 1):
       python -m eval.helica_eval.run_eval transcribe --engine nemo \
           --model nvidia/nemotron-3.5-asr-streaming-0.6b --set eval-d1 --chunk-ms 1120

Outputs go to runs/eval/<name>/: per_clip.csv and summary.json.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from .metrics import score, utterance_scores


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def score_files(ref_path: Path, hyp_path: Path, name: str, out_root: Path) -> dict:
    refs = {r["audio_filepath"]: r["text"] for r in _read_jsonl(ref_path)}
    hyps = {h["audio_filepath"]: h.get("pred_text", h.get("text", "")) for h in _read_jsonl(hyp_path)}

    missing = sorted(set(refs) - set(hyps))
    if missing:
        raise SystemExit(f"{len(missing)} reference clips have no hypothesis, e.g. {missing[:3]}")

    keys = sorted(refs)
    corpus = score([refs[k] for k in keys], [hyps[k] for k in keys])

    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "per_clip.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["audio_filepath", "cer", "wer", "ref", "hyp"])
        for k in keys:
            c, wr = utterance_scores(refs[k], hyps[k])
            w.writerow([k, f"{c:.4f}", f"{wr:.4f}", refs[k], hyps[k]])

    summary = {
        "name": name,
        "ref": str(ref_path),
        "hyp": str(hyp_path),
        "date": datetime.now().isoformat(timespec="seconds"),
        **corpus.as_dict(),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"per-clip results: {out_dir / 'per_clip.csv'}")
    return summary


def transcribe_and_score(args: argparse.Namespace) -> None:
    # TODO(Phase 1): implement engines.
    #  - "nemo": load the model with NeMo, run CACHE-AWARE STREAMING inference at --chunk-ms
    #    (use NeMo's streaming inference script/API for the installed version; check its --help),
    #    write hyps.jsonl, then call score_files().
    #  - "gguf": run NeMo-Speech.cpp on PC CPU with the same flags as the phone (parity check).
    #  - "whisper-ct2": ivrit.ai teacher baseline via faster-whisper.
    # Test set definitions: eval/test_sets.yaml. Always also run the hallucination suite.
    raise NotImplementedError("engine transcription not implemented yet — see TODO in run_eval.py")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("score", help="score an existing hypotheses file against a reference manifest")
    s.add_argument("--ref", type=Path, required=True)
    s.add_argument("--hyp", type=Path, required=True)
    s.add_argument("--name", required=True, help="run name, e.g. nemotron-zs@1120_eval-d1")
    s.add_argument("--out", type=Path, default=Path("runs/eval"))

    t = sub.add_parser("transcribe", help="transcribe a test set with an engine, then score (TODO)")
    t.add_argument("--engine", choices=["nemo", "gguf", "whisper-ct2"], required=True)
    t.add_argument("--model", required=True)
    t.add_argument("--set", required=True, help="test set name from eval/test_sets.yaml")
    t.add_argument("--chunk-ms", type=int, choices=[80, 160, 320, 560, 1120], default=1120)
    t.add_argument("--out", type=Path, default=Path("runs/eval"))

    args = p.parse_args()
    if args.cmd == "score":
        score_files(args.ref, args.hyp, args.name, args.out)
    else:
        transcribe_and_score(args)


if __name__ == "__main__":
    main()
