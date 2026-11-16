---
description: Log new test results into the docs (results file, plan checkboxes, CLAUDE.md status)
---

The user will paste or describe new test results: $ARGUMENTS

1. Decide which file in `docs/results/` they belong to (phone Nemotron, phone Whisper, PC eval, training run). If none fits, create `docs/results/<yyyy-mm-dd>_<topic>.md` using the same structure as the existing files (key findings → results tables → decisions → next steps).
2. Append the numbers using the existing table formats. Don't round away information. Mark pass/fail (✅ ❌ ⚠️).
3. If the results complete a checklist item, tick it in `docs/PLAN.md`; if they decide a gate, fill in the gate table.
4. If a decision follows, add it to `docs/DECISIONS.md`.
5. Update "Current status" in `CLAUDE.md` if the phase changed.
6. Finish with: a 3-line summary of what the results mean, and the next concrete step.
