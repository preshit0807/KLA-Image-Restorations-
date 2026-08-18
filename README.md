# ChipFinders — Semiconductor Image Restoration

**KLA SEMICON India Hackathon 2026 — PS1: AI-Based Restoration of Degraded Images for Semiconductor Inspection**

ChipFinders provides a compact, single-pass deep-learning model for restoring degraded grayscale semiconductor inspection images. The submission is packaged for the organizer's required offline inference interface and is designed to run directly on an NVIDIA GPU without internet access, API keys, additional model downloads, or manual model configuration.

## Submission Structure

```text
ChipFinders/
├── run.py
├── requirements.txt
├── README.md
└── models/
    ├── model_semiconrestore.py
    ├── model_semiconfreq_v2.py
    └── semiconrestore_freq_v2_best.pt
```

## Required Execution Command

From inside the `ChipFinders` directory:

```bash
python run.py <input-dir> <output-dir>
```

Example:

```bash
python run.py /path/to/input_npy /path/to/restored_output
```

Windows PowerShell example:

```powershell
python run.py `
"C:\path\to\input_npy" `
"C:\path\to\restored_output"
```

No checkpoint path, device flag, configuration file, or user interaction is required.

## Runtime Dependencies

Install the dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

The inference entry point uses only:

- NumPy
- PyTorch

A CUDA-enabled PyTorch installation and NVIDIA GPU are recommended for evaluation.

## Input Format

`run.py` reads every `.npy` file directly inside `<input-dir>`.

Expected input characteristics:

- grayscale image
- 2D array `(H, W)`; `(H, W, 1)` is also accepted
- floating-point values may lie outside `[0, 1]`
- input values are **not clipped before model feature extraction**
- all values must be finite

For the competition training data, the degraded input resolution is:

```text
128 × 128
```

## Output Format

For every input file:

```text
<input-dir>/example.npy
```

the script writes:

```text
<output-dir>/example.npy
```

with the same filename.

The produced restoration is:

- grayscale
- saved as `float32`
- finite: no `NaN` or `Inf`
- explicitly constrained to `[0, 1]`
- exactly `2×` the input height and width

For a `128 × 128` input, the output is:

```text
256 × 256
```

The output directory is created automatically if it does not already exist.

## Model

The submitted model is **SemiconRestore-Freq v2**.

### Architecture summary

The model combines:

- a degradation-aware NAFNet-style encoder/decoder backbone
- a residual bicubic reconstruction path
- four deterministic informational input channels:
  - raw intensity
  - `asinh`-compressed intensity
  - local morphological gradient
  - local variance
- a lightweight degradation encoder
- FiLM conditioning
- a small Haar frequency-detail adapter operating on high-resolution features

The frequency-detail adapter is intentionally lightweight and adds only a small number of parameters to the validated degradation-aware backbone.

### Final model size

```text
Parameters: 3,768,964
```

### Submitted checkpoint

```text
models/semiconrestore_freq_v2_best.pt
```

Checkpoint integrity:

```text
Size:   15,317,051 bytes
SHA256: 3128B29B1CD15772BFFB5D8CD120CCB0D1C622E655DA724E1AF684442DF611C0
```

The submitted checkpoint is the selected adapter-stage best model, not a later experimental `last.pt` checkpoint.

## Validation Results

The final model was evaluated on a deterministic 320-image held-out validation split.

| Metric | Result |
|---|---:|
| Mean PSNR | **28.429 dB** |
| Median PSNR | 28.611 dB |
| 5th-percentile PSNR | 20.587 dB |
| Mean SSIM | **0.7798** |
| Median SSIM | 0.8126 |
| Mean LPIPS | **0.2122** |

These values are internal held-out validation results and are not claims about unseen competition test data.

## Local Runtime Verification

The clean organizer package was tested end to end using the required command:

```bash
python run.py <input-dir> <output-dir>
```

A full 3,200-image local test completed successfully with:

```text
Inputs:              3200
Outputs:             3200
Bad outputs:         0
Output dtype:        float32
Value range:         [0, 1]
Filename mismatches: 0
```

Local runtime measurement from the clean organizer package:

```text
GPU:                   NVIDIA GeForce RTX 4050 Laptop GPU
Images:                3200
Batch size:            4
AMP FP16:              enabled
End-to-end time:       69.03 s
End-to-end throughput: 46.36 images/s
```

These measurements are hardware-specific. Organizer-side performance on an NVIDIA H100 may differ.

## Inference Behavior

At runtime, `run.py`:

1. locates the packaged checkpoint relative to the script
2. automatically selects CUDA when available
3. enables FP16 autocast on CUDA
4. chooses a conservative batch size from available GPU memory
5. loads all `.npy` inputs without requiring ground truth
6. preserves raw degraded values for the trained feature representation
7. performs single-pass 2× restoration
8. sanitizes and clamps outputs to `[0, 1]`
9. verifies the restored spatial resolution
10. saves one same-named `.npy` file per input

The inference code performs no network requests and downloads no external model files.

## Reproducibility and Safety Notes

- The model package contains all files required for inference.
- Ground-truth images are **not** required at runtime.
- No internet connection is required after dependencies are installed.
- No API key is required.
- No interactive prompt is used.
- No additional checkpoint download is performed.
- The submitted model sources correspond to the trained checkpoint architecture.
- Input data are not destructively clipped before learned restoration.
- Output validity is checked before saving.

## Quick Verification

To verify that the packaged checkpoint is the expected file:

### Windows PowerShell

```powershell
Get-Item .\models\semiconrestore_freq_v2_best.pt |
Select-Object FullName,Length

Get-FileHash .\models\semiconrestore_freq_v2_best.pt -Algorithm SHA256
```

Expected:

```text
Length = 15317051
SHA256 = 3128B29B1CD15772BFFB5D8CD120CCB0D1C622E655DA724E1AF684442DF611C0
```

## Team

**Team name:** ChipFinders  
**Challenge:** KLA SEMICON India Hackathon 2026 — PS1  
**Task:** AI-Based Restoration of Degraded Images for Semiconductor Inspection

