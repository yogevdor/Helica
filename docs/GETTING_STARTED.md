# Getting Started

## 0. Put the repo on GitHub (from Windows, once)

This folder was created on the Windows desktop. Push it to GitHub, then clone it on Fedora.

1. Choose a repo license (see `docs/DECISIONS.md` → Open decisions) and add a `LICENSE` file.
2. GitHub Desktop → *File → Add local repository* → choose this folder → *create a repository* → *Publish repository* (public, name `helica`).
   Or with git:
   ```bash
   cd helica
   git init -b main
   git add .
   git commit -m "Initial commit: plan, architecture, results, scaffolding"
   git remote add origin https://github.com/<you>/helica.git
   git push -u origin main
   ```
3. `.gitattributes` forces LF line endings so the shell scripts work on Fedora and Termux.

## 1. Fedora training machine

```bash
# NVIDIA driver (RPM Fusion). Blackwell needs open kernel modules (driver 570+).
sudo dnf install https://mirrors.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm \
                 https://mirrors.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm -E %fedora).noarch.rpm
sudo dnf install akmod-nvidia xorg-x11-drv-nvidia-cuda
# Secure Boot on? Enroll the akmods key (see RPM Fusion's NVIDIA howto), then reboot.
nvidia-smi        # must show RTX 5060 Ti and driver 570+

# System tools
sudo dnf install git git-lfs ffmpeg sox gcc-c++ cmake ninja-build python3.11

# Python env
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv ~/asr-env --python 3.11
source ~/asr-env/bin/activate

# PyTorch for Blackwell (CUDA 12.8 wheels)
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability())"
# expect: True (12, 0)

# NeMo + repo requirements
uv pip install "nemo_toolkit[asr]"
uv pip install -e ".[dev]"
```

### Data drive

```bash
# NVMe only: e.g. /data = a folder on the NVMe. With the SATA drive: mount it at /data
echo 'export HF_HOME=/data/hf' >> ~/.bashrc
echo 'export HELICA_DATA=/data/helica' >> ~/.bashrc
source ~/.bashrc
```
Set these **before** the first dataset download, so you always know where the data goes. Check free space with `df -h` before big downloads.

### Smoke tests

```bash
python -m eval.helica_eval.metrics          # self-test of CER/WER + Hebrew normalization
python -m training.scripts.build_manifests --help
```

## 2. Claude Code

```bash
npm install -g @anthropic-ai/claude-code   # or see Anthropic docs for the current install method
cd ~/helica
claude
```
Claude Code reads `CLAUDE.md` automatically. Good first prompts:
- "Read docs/PLAN.md Phase 1 and tell me what's next."
- "Help me finish eval/helica_eval/run_eval.py for Nemotron zero-shot on eval-d1."
- `/log-result` (custom command in `.claude/commands/`) after a test session.

## 3. Phone (Termux)

See `phone-bench/README.md`. Build scripts and heat-test loops for whisper.cpp and NeMo-Speech.cpp are there.
