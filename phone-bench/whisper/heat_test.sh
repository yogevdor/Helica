#!/data/data/com.termux/files/usr/bin/bash
# ~11-minute sustained-load test (600 runs). Run from ~/whisper.cpp with run.sh next to it.
B=models/ggml-base-q8_0.bin
V="--vad -vm models/ggml-silero-v5.1.2.bin"
termux-wake-lock
for i in $(seq 1 600); do
  echo "$i $(bash run.sh $B clip7s.wav 512 $V | grep -o 'total time = *[0-9.]*')"
done | tee heat_10min.txt
head -5 heat_10min.txt; tail -5 heat_10min.txt
