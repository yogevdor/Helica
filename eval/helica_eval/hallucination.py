"""Hallucination suite: non-speech clips must produce NO text.

This is a hard gate for every model (after every training run and after every export/quantization).
A deaf user cannot tell invented text from real speech, so any output here is a failure,
regardless of accuracy numbers.

The check uses the RAW output (only whitespace stripped), not the scoring normalization: a lone "." or
"?" on silence is still text on the user's screen.
"""

from __future__ import annotations

from dataclasses import dataclass

# Clip names match the phone tests (see docs/results/). Audio lives outside the repo, under $HELICA_DATA.
NON_SPEECH_CLIPS = [
    "silence_digital.wav",  # generated: ffmpeg anullsrc, 10 s
    "silence10s.wav",  # recorded room silence, 10 s
    "noise_pink.wav",  # generated pink noise, 10 s
    "tvnoise10s.wav",  # recorded TV noise, 10 s
]


@dataclass
class HallucinationResult:
    clip: str
    output: str

    @property
    def passed(self) -> bool:
        return self.output.strip() == ""


def check(outputs: dict[str, str]) -> list[HallucinationResult]:
    """outputs: clip name → transcript produced by the model."""
    missing = [c for c in NON_SPEECH_CLIPS if c not in outputs]
    if missing:
        raise ValueError(f"missing hallucination clips: {missing}")
    return [HallucinationResult(c, outputs[c]) for c in NON_SPEECH_CLIPS]


def report(results: list[HallucinationResult]) -> str:
    lines = []
    for r in results:
        mark = "✅ nothing" if r.passed else f"❌ {r.output!r}"
        lines.append(f"| {r.clip} | {mark} |")
    verdict = "PASS" if all(r.passed for r in results) else "FAIL"
    return "| Clip | Output |\n|---|---|\n" + "\n".join(lines) + f"\n\nHallucination suite: **{verdict}**"
