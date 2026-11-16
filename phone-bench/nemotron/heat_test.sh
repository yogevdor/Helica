#!/data/data/com.termux/files/usr/bin/bash
# 20-run streaming heat test, 1120 ms chunks (R=13). Run from ~/NeMo-Speech.cpp.
# Clips: clips/long.wav = 66 s (ffmpeg -stream_loop 5 -i clips/jfk.wav -ar 16000 -ac 1 clips/long.wav)
# Usage: bash heat_test.sh [R] [LANG] [CLIP] [RUNS]
# Gate A was measured with the English defaults below. For Gate A′ use Hebrew, e.g. a ~60 s clip × 60 runs:
#   bash heat_test.sh 13 he-IL clips/hebrew_long.wav 60     (screen on, unplugged, cooled first)
R=${1:-13}; LANG_TAG=${2:-en-US}; CLIP=${3:-clips/long.wav}; RUNS=${4:-20}
NS=~/NeMo-Speech.cpp/build/cpu-asr/bin/nemo-speech
termux-wake-lock
echo "battery before: $(termux-battery-status 2>/dev/null | grep percentage || echo 'n/a')"
for i in $(seq 1 "$RUNS"); do
  s=$(date +%s%N)
  $NS transcribe "$CLIP" -l "$LANG_TAG" --stream --asr.streaming.rnnt_right_context "$R" > /dev/null 2>&1
  e=$(date +%s%N)
  echo "$i $(( (e - s) / 1000000 )) ms"
done | tee "heat_nemo_R${R}.txt"
echo "battery after: $(termux-battery-status 2>/dev/null | grep percentage || echo 'n/a')"
