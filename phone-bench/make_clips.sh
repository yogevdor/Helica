#!/data/data/com.termux/files/usr/bin/bash
# Create synthetic test clips and convert phone recordings. Run from ~/whisper.cpp.
set -euo pipefail
mkdir -p clips

# once: termux-setup-storage   (recordings are then in ~/storage/downloads)
# convert a recording:  ffmpeg -i ~/storage/downloads/NAME.m4a -ar 16000 -ac 1 clips/NAME.wav

ffmpeg -y -f lavfi -i anullsrc=r=16000:cl=mono -t 10 clips/silence_digital.wav
ffmpeg -y -f lavfi -i "anoisesrc=color=pink:amplitude=0.05:r=16000" -t 10 -ac 1 clips/noise_pink.wav

# noisy versions of the clean clip (noise file must be at least as long as the speech)
ffmpeg -y -i clip7s.wav -i clips/noise_pink.wav \
  -filter_complex "[1]volume=0.3[n];[0][n]amix=inputs=2:duration=first" -ar 16000 -ac 1 clips/noisy_mild.wav
ffmpeg -y -i clip7s.wav -i clips/noise_pink.wav \
  -filter_complex "[1]volume=1.0[n];[0][n]amix=inputs=2:duration=first" -ar 16000 -ac 1 clips/noisy_hard.wav
