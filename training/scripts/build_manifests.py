"""Hugging Face audio dataset → NeMo manifest (JSON lines) + 16 kHz mono FLAC files.

Example:
    python -m training.scripts.build_manifests \
        --dataset ivrit-ai/<dataset> --split train --max-hours 150 \
        --audio-col audio --text-col text --id-col id --source-col <episode-column> \
        --max-hours-per-source 2 --val-frac 0.02 \
        --out $HELICA_DATA/pilot --exclude eval/test_sets.yaml

Column names differ between datasets: check the dataset card first and pass them explicitly.

Leakage guard: any sample whose id or --source-col value starts with an `exclude_sources` entry in
eval/test_sets.yaml is skipped and counted. The script refuses to run
  - while that list still contains a TODO placeholder (unless --allow-todo-exclude), and
  - when the list is non-empty but no --source-col is given, because then the check would only
    compare row ids and silently exclude nothing (unless --allow-no-source, for corpora that
    cannot contain eval audio, e.g. English data).
List whole podcast series / speakers in `exclude_sources`, not just the eval episode.

Sampling: rows are shuffled (--seed) so --max-hours doesn't just take the first episodes in
dataset order. --val-frac sends whole sources (episodes) to val_manifest.jsonl, chosen by a
stable hash of the source id, so validation never shares an episode with training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml
from tqdm import tqdm

TARGET_SR = 16_000


def load_excludes(path: Path, allow_todo: bool) -> set[str]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    ex = {str(x) for x in (cfg.get("exclude_sources") or [])}
    if "TODO" in ex and not allow_todo:
        raise SystemExit(
            f"{path}: exclude_sources still contains TODO. Fill in the eval-d1 podcast series id(s) first "
            "(or pass --allow-todo-exclude for a dry run on data that cannot contain eval audio)."
        )
    ex.discard("TODO")
    return ex


def is_excluded(sid: str, src: str, excludes: set[str]) -> bool:
    """True if the sample id or source id starts with any `exclude_sources` entry.

    Entries are prefixes (e.g. `Mindset/2018.10.07` matches every clip uuid of that episode), the same
    rule as `select_crowd_v5.py`. A full id is just a prefix of itself, so exact ids still work.
    """
    return any(x.startswith(p) for p in excludes for x in (sid, src))


def to_mono_16k(array: np.ndarray, sr: int) -> np.ndarray:
    if array.ndim > 1:
        array = array.mean(axis=-1) if array.shape[-1] <= 8 else array.mean(axis=0)
    if sr != TARGET_SR:
        import librosa  # only needed if the dataset isn't already 16 kHz

        array = librosa.resample(array.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR)
    return array.astype(np.float32)


def is_val_source(source_id: str, val_frac: float) -> bool:
    """Stable across runs and machines (unlike Python's hash())."""
    h = int(hashlib.md5(source_id.encode("utf-8")).hexdigest(), 16)
    return h / 16**32 < val_frac


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--split", default="train")
    p.add_argument("--audio-col", default="audio")
    p.add_argument("--text-col", default="text")
    p.add_argument("--id-col", default=None, help="column with a unique sample id (else row index)")
    p.add_argument("--source-col", default=None, help="episode/source id column, checked against excludes")
    p.add_argument("--lang", default="he-IL")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--exclude", type=Path, default=Path("eval/test_sets.yaml"))
    p.add_argument("--allow-todo-exclude", action="store_true")
    p.add_argument(
        "--allow-no-source",
        action="store_true",
        help="run without --source-col although excludes are set (only for data with no eval audio)",
    )
    p.add_argument("--max-hours", type=float, default=None, help="training hours budget (val not counted)")
    p.add_argument("--max-hours-per-source", type=float, default=None, help="cap per episode/source")
    p.add_argument("--val-frac", type=float, default=0.0, help="fraction of SOURCES held out as validation")
    p.add_argument("--seed", type=int, default=0, help="shuffle seed")
    p.add_argument("--no-shuffle", action="store_true", help="keep dataset order (bad with --max-hours)")
    p.add_argument("--min-sec", type=float, default=0.5)
    p.add_argument("--max-sec", type=float, default=20.0, help="cap per sample (16 GB VRAM budget)")
    p.add_argument("--streaming", action="store_true", help="stream from HF instead of downloading the split")
    args = p.parse_args()

    excludes = load_excludes(args.exclude, args.allow_todo_exclude)
    if excludes and not args.source_col and not args.allow_no_source:
        raise SystemExit(
            f"{args.exclude}: exclude_sources is set, but no --source-col was given, so only row ids "
            "would be checked and the eval episode/series would NOT be excluded. Pass --source-col "
            "(or --allow-no-source for data that cannot contain eval audio)."
        )
    if (args.val_frac or args.max_hours_per_source) and not args.source_col:
        raise SystemExit("--val-frac and --max-hours-per-source need --source-col (they work per source)")

    from datasets import load_dataset  # imported here so --help works without `datasets`

    audio_dir = args.out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / f"{args.split}_manifest.jsonl"
    val_path = args.out / "val_manifest.jsonl"

    ds = load_dataset(args.dataset, args.config, split=args.split, streaming=args.streaming)
    if not args.no_shuffle:
        # Streaming datasets shuffle within a buffer only: still biased toward dataset order.
        ds = ds.shuffle(seed=args.seed, buffer_size=10_000) if args.streaming else ds.shuffle(seed=args.seed)

    kept = kept_val = skipped_len = skipped_text = skipped_excl = skipped_cap = 0
    total_sec = val_sec = 0.0
    per_source_sec: dict[str, float] = defaultdict(float)
    budget = args.max_hours * 3600 if args.max_hours else None
    source_cap = args.max_hours_per_source * 3600 if args.max_hours_per_source else None

    with ExitStack() as stack:
        mf = stack.enter_context(manifest_path.open("w", encoding="utf-8"))
        # only (over)write the val manifest when this run actually builds one
        vf = stack.enter_context(val_path.open("w", encoding="utf-8")) if args.val_frac > 0 else None
        for i, row in enumerate(tqdm(ds, desc="samples")):
            sid = str(row[args.id_col]) if args.id_col else f"{i:09d}"
            src = str(row[args.source_col]) if args.source_col else sid
            if is_excluded(sid, src, excludes):
                skipped_excl += 1
                continue
            to_val = args.val_frac > 0 and is_val_source(src, args.val_frac)
            if source_cap and not to_val and per_source_sec[src] >= source_cap:
                skipped_cap += 1
                continue

            text = (row.get(args.text_col) or "").strip()
            if not text:
                skipped_text += 1
                continue

            audio = row[args.audio_col]
            wav = to_mono_16k(np.asarray(audio["array"]), int(audio["sampling_rate"]))
            dur = len(wav) / TARGET_SR
            if not (args.min_sec <= dur <= args.max_sec):
                skipped_len += 1
                continue

            safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in sid)
            path = audio_dir / f"{safe}.flac"
            sf.write(path, wav, TARGET_SR, format="FLAC")
            entry = {"audio_filepath": str(path.resolve()), "duration": round(dur, 3), "text": text}
            line = json.dumps({**entry, "lang": args.lang}, ensure_ascii=False)
            if vf is not None and to_val:
                vf.write(line + "\n")
                kept_val += 1
                val_sec += dur
                continue
            mf.write(line + "\n")
            kept += 1
            total_sec += dur
            per_source_sec[src] += dur
            if budget and total_sec >= budget:
                break

    print(
        f"manifest: {manifest_path}\n"
        f"kept {kept} samples, {total_sec / 3600:.1f} h from {len(per_source_sec)} sources | "
        f"val: {kept_val} samples, {val_sec / 3600:.1f} h ({val_path}) | skipped: excluded={skipped_excl}, "
        f"empty_text={skipped_text}, out_of_length={skipped_len}, source_cap={skipped_cap}"
    )


if __name__ == "__main__":
    main()
