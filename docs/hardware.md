# Hardware

## Training PC (Fedora)

| Part | Model |
|---|---|
| CPU | AMD Ryzen 5 7500F (AM5, 6 cores, no iGPU) |
| Motherboard | ASRock B650M-HDV/M.2 |
| GPU | MSI GeForce RTX 5060 Ti 16G Shadow 2X OC (16 GB GDDR7, 128-bit, Blackwell, compute capability 12.0) |
| RAM | Crucial Pro 2×16 GB DDR5-6000 CL36 (EXPO) |
| Storage | 500 GB NVMe (OS, code, active datasets); optional 500 GB–1 TB SATA for HF cache and archives (D-019) |
| PSU | Antec GSK 750W ATX 3.1, 80+ Gold, fully modular |
| Case | Lian-Li A3 Black |
| Thermal paste | Noctua NT-H2 AM5 Edition |

Notes:
- Blackwell needs CUDA 12.8+ and a PyTorch build from the `cu128` index; NVIDIA open kernel modules (driver 570+).
- Enable EXPO in BIOS so RAM runs at 6000 MT/s.
- 16 GB VRAM: bf16, small batches + gradient accumulation, capped sample duration.

## Test phone

| | |
|---|---|
| Model | Poco X4 Pro 5G |
| SoC | Snapdragon 695 (2× Cortex-A78 + 6× Cortex-A55) |
| RAM | ~7.3 GiB visible |
| CPU features (ggml) | dotprod ✅, FP16 arithmetic ✅, i8mm ❌, SVE ❌ |
| Tools | Termux (F-Droid) |
