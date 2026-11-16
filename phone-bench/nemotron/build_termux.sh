#!/data/data/com.termux/files/usr/bin/bash
# Build NeMo-Speech.cpp (CPU ASR) in Termux, including every fix found during Gate A.
# See docs/results/gate_a_nemotron_phone_tests.md for the problems these fixes solve.
set -euo pipefail

pkg update
pkg install -y git cmake ninja clang pkg-config ffmpeg

# SentencePiece v0.2.0: newer versions pull in system abseil (C++20) and break the C++17 build
cd ~
[ -d sentencepiece ] || git clone --depth 1 --branch v0.2.0 https://github.com/google/sentencepiece
cd sentencepiece
cmake -B build -G Ninja -DCMAKE_INSTALL_PREFIX="$PREFIX" -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build build -j4
cmake --install build

# NeMo-Speech.cpp. Pin a commit for reproducible results: NS_REF=<commit or tag> bash build_termux.sh
NS_REF=${NS_REF:-}
cd ~
[ -d NeMo-Speech.cpp ] || git clone https://github.com/NVIDIA/NeMo-Speech.cpp
cd NeMo-Speech.cpp
if [ -n "$NS_REF" ]; then git fetch --tags origin && git checkout "$NS_REF"; fi
git submodule update --init ggml
git submodule update --init --recursive llama.cpp
NS_COMMIT=$(git rev-parse HEAD)
echo "$NS_COMMIT" > ~/ns_build_commit.txt
echo "NeMo-Speech.cpp commit: $NS_COMMIT (saved to ~/ns_build_commit.txt; put it in the results log)"

# Make thread count adjustable via NS_THREADS (default stays 4, the fastest on Snapdragon 695).
sed -i 's/cr.gf, 4))/cr.gf, std::getenv("NS_THREADS") ? std::atoi(std::getenv("NS_THREADS")) : 4))/' \
  src/runtime/ggml/session.cpp
if ! grep -q 'NS_THREADS' src/runtime/ggml/session.cpp; then
  echo "NS_THREADS patch did not apply: the line moved in this version. Patch src/runtime/ggml/session.cpp by hand." >&2
  exit 1
fi

termux-wake-lock
# -llog: fixes "cannot locate symbol __android_log_write"
scripts/configure.sh cpu-asr -DCMAKE_SHARED_LINKER_FLAGS="-llog" -DCMAKE_EXE_LINKER_FLAGS="-llog"
cmake --build --preset cpu-asr -j4

grep -q "alias ns=" ~/.bashrc || echo "alias ns=$HOME/NeMo-Speech.cpp/build/cpu-asr/bin/nemo-speech" >> ~/.bashrc
"$HOME/NeMo-Speech.cpp/build/cpu-asr/bin/nemo-speech" doctor
echo "Next: ns pull nemotron-3.5   (~650 MB)"
