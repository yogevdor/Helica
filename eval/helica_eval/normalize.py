"""Hebrew text normalization for CER/WER.

The SAME function must be applied to references and hypotheses.

STATUS: provisional. Phase 1 replaces this with (or aligns it to) the normalization used by the
ivrit.ai Hebrew transcription leaderboard, so our numbers are comparable. The leaderboard links to
`evaluate_model.py` in github.com/ivrit-ai/asr-training as its evaluation script. After aligning, the
first check is reproducing Nemotron zero-shot eval-d1 ≈ 0.171 (that number is WER).
"""

from __future__ import annotations

import re
import unicodedata

# Hebrew cantillation marks and niqqud (vowel points): U+0591–U+05C7, except the letters themselves.
# Maqaf (U+05BE, Hebrew hyphen) is turned into a space before removal.
_HEB_MARKS = re.compile(r"[֑-ֽֿ-ׇׂׅׄ]")
_MAQAF = "־"
# Geresh / gershayim (׳ ״) and their ASCII look-alikes are dropped (e.g. צ'יפס → ציפס, צה"ל → צהל).
_QUOTES = re.compile(r"[׳״'\"`´‘’“”„]")
# Everything that is not a letter, digit, or whitespace becomes a space.
_NON_WORD = re.compile(r"[^\w\s]", flags=re.UNICODE)
_SPACES = re.compile(r"\s+")


def normalize_hebrew(text: str) -> str:
    """Normalize a Hebrew (+ English) transcript for scoring.

    - Unicode NFKC
    - remove niqqud and cantillation
    - maqaf → space; geresh/gershayim/quotes removed
    - punctuation → space
    - Latin letters lower-cased
    - whitespace collapsed
    Final letters (ך ם ן ף ץ) are kept as written.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace(_MAQAF, " ")
    text = _HEB_MARKS.sub("", text)
    text = _QUOTES.sub("", text)
    text = _NON_WORD.sub(" ", text)
    text = text.replace("_", " ")
    text = text.lower()
    return _SPACES.sub(" ", text).strip()
