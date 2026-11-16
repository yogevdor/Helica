# Phone benchmarks (Termux)

Scripts used for the phone tests in `docs/results/`. Run them in Termux on the test phone.

**Rules:** phone unplugged, cooled ~10 min, screen off between runs if possible, `termux-wake-lock` for long runs, record battery % before and after.

| Path | What |
|---|---|
| `make_clips.sh` | Synthetic test clips (silence, pink noise, noisy speech) + converting phone recordings |
| `whisper/run.sh` | whisper.cpp wrapper: `bash run.sh MODEL CLIP AC [extra flags]` |
| `whisper/heat_test.sh` | 600-run (~11 min) heat test |
| `nemotron/build_termux.sh` | Full NeMo-Speech.cpp build in Termux, with all fixes |
| `nemotron/heat_test.sh` | 20-run streaming heat test (1120 ms chunks) |

RTF (Nemotron) = (processing time − ~4 s startup) ÷ audio length.
