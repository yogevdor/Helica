"""Select the crowd-transcribe-v5 clips to train on, following decision D-022 (docs/DECISIONS.md).

Input:  data/crowd_v5_meta.parquet  - every column except audio, made once with DuckDB
        eval/test_sets.yaml          - `exclude_sources`: uuid prefixes that must never be used
Output: data/crowd_v5_<split>_selected.jsonl - one line per kept clip: {"uuid", "text", "duration"}

Stage 2 (on the Fedora PC) reads these files, downloads the audio of the listed clips, and writes the
NeMo manifest. All the filtering decisions live here, so they can be checked without any audio.

Usage (from the repo root, with the .venv active):
    python -m training.scripts.select_crowd_v5               # both splits, max 30 s
    python -m training.scripts.select_crowd_v5 --max-sec 20  # if the GPU memory test fails at 30 s

The rules (D-020, D-021, D-022), in the order they are applied:
  1. text = `sentence` with spaces trimmed; drop empty text
  2. drop rows marked `unintelligible` or `foreign_language` by the volunteer
  3. a clip (uuid) with several transcripts: keep one if they all have the same words
     (ignoring punctuation, spaces and Latin upper/lower case), otherwise drop the clip
  4. drop clips longer than --max-sec
  5. drop clips whose uuid starts with an `exclude_sources` prefix (eval-d1 leakage guard)
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml

SPLITS = ("train", "test")


@dataclass
class Row:
    """One row of the metadata file: one transcript of one clip by one volunteer."""

    uuid: str
    sentence: str | None
    duration: float | None
    unintelligible: bool
    foreign_language: bool


def load_rows(meta_path: Path, split: str) -> list[Row]:
    """Read the rows of one split from the local metadata parquet file.

    DuckDB is used only as a fast parquet reader here: it returns plain tuples, and every rule is
    applied in Python below. `coalesce(x, false)` turns a missing flag (NULL) into False.
    """
    import duckdb  # imported here so the selection logic can be tested without DuckDB

    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    query = f"""
        SELECT uuid,
               sentence,
               extra_data.duration,
               coalesce(extra_data.unintelligible, false),
               coalesce(extra_data.foreign_language, false)
        FROM read_parquet('{meta_path.as_posix()}')
        WHERE split = '{split}'
    """
    return [Row(*t) for t in duckdb.connect().sql(query).fetchall()]


def load_exclude_prefixes(test_sets_path: Path) -> list[str]:
    """`exclude_sources` from eval/test_sets.yaml. Refuses to run while it still says TODO."""
    cfg = yaml.safe_load(test_sets_path.read_text(encoding="utf-8"))
    prefixes = [str(p) for p in (cfg.get("exclude_sources") or [])]
    if not prefixes or "TODO" in prefixes:
        raise SystemExit(f"{test_sets_path}: fill in `exclude_sources` first (eval-d1 leakage guard).")
    return prefixes


def words_only(text: str) -> str:
    """Reduce a transcript to its words, so two transcripts can be compared fairly.

    Deletes every character that is not a letter, digit or space (punctuation, gershayim, niqqud),
    lower-cases Latin letters, and collapses runs of spaces. Two volunteers who wrote the same words
    with different punctuation get the same result.
    """
    kept = "".join(ch for ch in text if ch.isalnum() or ch.isspace())
    return " ".join(kept.lower().split())


def select_clips(
    rows: list[Row], max_sec: float, exclude_prefixes: list[str]
) -> tuple[list[dict], dict[str, float]]:
    """Apply rules 1-5. Returns (kept clips, statistics of what each rule removed)."""
    stats: Counter[str] = Counter(rows_in=len(rows))

    # Rules 1-2: per-row checks
    usable: dict[str, list[Row]] = defaultdict(list)  # uuid -> its usable transcripts
    for row in rows:
        text = (row.sentence or "").strip()
        if not text:
            stats["drop_row_empty_text"] += 1
        elif row.unintelligible:
            stats["drop_row_unintelligible"] += 1
        elif row.foreign_language:
            stats["drop_row_foreign_language"] += 1
        elif row.duration is None:
            stats["drop_row_no_duration"] += 1
        else:
            usable[row.uuid].append(Row(row.uuid, text, row.duration, False, False))

    stats["clips_with_usable_text"] = len(usable)

    kept: list[dict] = []
    for uuid, versions in usable.items():
        # Rule 3: several transcripts of the same clip must agree on the words
        if len(versions) > 1:
            stats["clips_with_2plus_transcripts"] += 1
            if len({words_only(v.sentence) for v in versions}) > 1:
                stats["drop_clip_transcripts_disagree"] += 1
                continue
        duration = versions[0].duration
        # Rule 4: length limit
        if duration > max_sec:
            stats["drop_clip_too_long"] += 1
            stats["drop_clip_too_long_hours"] += duration / 3600
            continue
        # Rule 5: leakage guard
        if any(uuid.startswith(p) for p in exclude_prefixes):
            stats["drop_clip_excluded_source"] += 1
            continue
        # Same words: keep the most common spelling; ties broken alphabetically, so reruns match.
        spellings = Counter(v.sentence for v in versions)
        text = min(spellings, key=lambda s: (-spellings[s], s))
        kept.append({"uuid": uuid, "text": text, "duration": duration})

    kept.sort(key=lambda c: c["uuid"])
    stats["kept_clips"] = len(kept)
    stats["kept_hours"] = sum(c["duration"] for c in kept) / 3600
    return kept, dict(stats)


def series_report(kept: list[dict], exclude_prefixes: list[str]) -> dict[str, tuple[int, float]]:
    """For each excluded prefix: how many KEPT clips share its podcast series (text before the first '/').

    Information only, for the open question of excluding a whole series, not just one episode.
    """
    report = {}
    for prefix in exclude_prefixes:
        series = prefix.split("/")[0] + "/"
        same = [c for c in kept if c["uuid"].startswith(series)]
        report[series] = (len(same), sum(c["duration"] for c in same) / 3600)
    return report


def write_jsonl(clips: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for clip in clips:
            f.write(json.dumps(clip, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--meta", type=Path, default=Path("data/crowd_v5_meta.parquet"))
    p.add_argument("--exclude", type=Path, default=Path("eval/test_sets.yaml"))
    p.add_argument("--split", choices=[*SPLITS, "both"], default="both")
    p.add_argument("--max-sec", type=float, default=30.0, help="D-022: 30 if memory allows, else 20")
    p.add_argument("--out-dir", type=Path, default=Path("data"))
    args = p.parse_args()

    prefixes = load_exclude_prefixes(args.exclude)
    splits = SPLITS if args.split == "both" else (args.split,)
    for split in splits:
        rows = load_rows(args.meta, split)
        kept, stats = select_clips(rows, args.max_sec, prefixes)
        out = args.out_dir / f"crowd_v5_{split}_selected.jsonl"
        write_jsonl(kept, out)

        print(f"\n=== {split} (max {args.max_sec:g} s) -> {out}")
        for key, value in stats.items():
            print(f"  {key:32s} {value:12,.1f}" if isinstance(value, float) else f"  {key:32s} {value:12,}")
        for series, (n, hours) in series_report(kept, prefixes).items():
            print(f"  kept clips from series {series!r}: {n:,} ({hours:.1f} h)")


if __name__ == "__main__":
    main()
