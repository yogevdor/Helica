"""CER / WER for Hebrew, using one shared normalization.

Run `python -m eval.helica_eval.metrics` for a quick self-test.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import jiwer

from .normalize import normalize_hebrew


@dataclass
class Scores:
    cer: float
    wer: float
    n_utts: int
    n_ref_words: int
    n_ref_chars: int
    n_empty_ref: int = 0  # utterances with an empty reference (not in CER/WER)
    n_empty_ref_with_text: int = 0  # ...where the model still produced text (insertions on non-speech)

    def as_dict(self) -> dict[str, float | int]:
        return {
            "cer": round(self.cer, 4),
            "wer": round(self.wer, 4),
            "n_utts": self.n_utts,
            "n_ref_words": self.n_ref_words,
            "n_ref_chars": self.n_ref_chars,
            "n_empty_ref": self.n_empty_ref,
            "n_empty_ref_with_text": self.n_empty_ref_with_text,
        }


def score(refs: Sequence[str], hyps: Sequence[str], normalize: bool = True) -> Scores:
    """Corpus-level CER and WER (errors summed over all utterances, not averaged per utterance).

    Empty references are left out of CER/WER, but counted: any text on them (raw, not normalized)
    is an insertion on non-speech and is reported as `n_empty_ref_with_text`.
    """
    if len(refs) != len(hyps):
        raise ValueError(f"refs ({len(refs)}) and hyps ({len(hyps)}) differ in length")

    pairs = []
    n_empty = n_empty_with_text = 0
    for r, h in zip(refs, hyps, strict=True):
        r_n = normalize_hebrew(r) if normalize else r
        h_n = normalize_hebrew(h) if normalize else h
        if r_n:
            pairs.append((r_n, h_n))
        else:
            n_empty += 1
            n_empty_with_text += bool(h.strip())
    if not pairs:
        raise ValueError("no non-empty references to score")

    r_list = [p[0] for p in pairs]
    h_list = [p[1] for p in pairs]
    wer = jiwer.wer(r_list, h_list)
    # CER on characters without spaces would hide word-boundary errors; keep spaces.
    cer = jiwer.cer(r_list, h_list)
    return Scores(
        cer=cer,
        wer=wer,
        n_utts=len(pairs),
        n_ref_words=sum(len(r.split()) for r in r_list),
        n_ref_chars=sum(len(r) for r in r_list),
        n_empty_ref=n_empty,
        n_empty_ref_with_text=n_empty_with_text,
    )


def utterance_scores(ref: str, hyp: str) -> tuple[float, float]:
    """(CER, WER) for a single utterance, for per-clip CSV logging."""
    r, h = normalize_hebrew(ref), normalize_hebrew(hyp)
    if not r:
        return (0.0 if not h else 1.0), (0.0 if not h else 1.0)
    return jiwer.cer(r, h), jiwer.wer(r, h)


def _self_test() -> None:
    assert normalize_hebrew("שָׁלוֹם, עוֹלָם!") == "שלום עולם"
    assert normalize_hebrew('צה"ל בתל־אביב') == "צהל בתל אביב"
    assert normalize_hebrew("Deadline של ה-Report") == "deadline של ה report"
    s = score(["שלום עולם"], ["שלום עולם"])
    assert s.cer == 0 and s.wer == 0
    s = score(["שלום עולם"], ["שלום"])
    assert s.wer == 0.5
    s = score(["שלום", ""], ["שלום", "."])
    assert s.n_empty_ref == 1 and s.n_empty_ref_with_text == 1
    from .hallucination import HallucinationResult

    assert not HallucinationResult("silence_digital.wav", ".").passed
    assert HallucinationResult("silence_digital.wav", "  ").passed
    print("metrics self-test OK:", score(["אני הולך הביתה"], ["אני הלך הביתה"]).as_dict())


if __name__ == "__main__":
    _self_test()
