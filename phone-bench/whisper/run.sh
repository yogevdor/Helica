#!/data/data/com.termux/files/usr/bin/bash
# usage (from ~/whisper.cpp): bash run.sh MODEL CLIP AC [extra flags]
#   B=models/ggml-base-q8_0.bin
#   V="--vad -vm models/ggml-silero-v5.1.2.bin"     # get it with ./models/download-vad-model.sh silero-v5.1.2
#   bash run.sh $B clips/silence_digital.wav 512 $V
M=$1; F=$2; AC=$3; shift 3
./build/bin/whisper-cli -m "$M" -f "$F" -l he -t 6 -ac "$AC" -bs 1 -bo 1 "$@" 2>&1 \
  | grep -aE "^\[|encode time|decode time|total time"
