# ChipFinders — KLA SEMICON India Hackathon 2026 PS1

Minimal offline inference submission for **AI-Based Restoration of Degraded Images for Semiconductor Inspection**.

## Required structure

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

The checkpoint is included directly in `models/`; no network access or additional model download is required.

## Setup

Use a Python environment with an NVIDIA-compatible PyTorch installation, then install the declared runtime dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run

The evaluator-facing command is exactly:

```bash
python run.py <input-dir> <output-dir>
```

Example:

```bash
python run.py ./input ./output
```

No additional flags, API keys, downloads, prompts, or manual model configuration are required.

## Input contract

- `input-dir` contains degraded grayscale `.npy` files.
- Files are read directly; no PNG/JPEG conversion is required.
- Standard competition inputs are 128×128 grayscale arrays.
- Raw input values are not clipped before restoration.

## Output contract

For every input `.npy` file, `run.py`:

- creates one output `.npy` file;
- preserves the exact filename;
- writes a 2D grayscale `float32` array;
- restores at exactly 2× spatial resolution (128×128 → 256×256 for standard inputs);
- guarantees finite output values;
- clamps output values to `[0, 1]`;
- creates `output-dir` automatically if it does not exist.

## Model

The runtime model is **SemiconRestore-Freq v2**. `run.py` loads the local checkpoint:

```text
models/semiconrestore_freq_v2_best.pt
```

The model runs locally with PyTorch and automatically uses CUDA when available.
