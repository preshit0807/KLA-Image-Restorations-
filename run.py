#!/usr/bin/env python3
"""KLA SEMICON 2026 final inference entry point.

Required organizer interface:
    python run.py <input-dir> <output-dir>

The script is self-contained, performs no network access, reads every .npy file
from the input directory, and writes one restored .npy file with the same name
into the output directory.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# Keep the submission folder clean when the evaluator imports local model code.
sys.dont_write_bytecode = True
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
CHECKPOINT = MODEL_DIR / "semiconrestore_freq_v2_best.pt"

# The model file uses a top-level import of model_semiconrestore.py.  Adding the
# model directory to sys.path keeps the submitted model sources unchanged.
sys.path.insert(0, str(MODEL_DIR))
from model_semiconfreq_v2 import SemiconRestoreFreqV2  # noqa: E402


def _load_checkpoint(path: Path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {path}. "
            "The final submission must include models/semiconrestore_freq_v2_best.pt"
        )

    # A Git LFS pointer is a tiny text file, not the real checkpoint.  Catch it
    # explicitly so an incomplete clone cannot silently reach evaluation.
    if path.stat().st_size < 1_000_000:
        try:
            prefix = path.read_bytes()[:80]
        except OSError:
            prefix = b""
        if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise RuntimeError(
                f"{path} is a Git LFS pointer, not the model weights. "
                "Run `git lfs pull` before creating the offline submission package."
            )
        raise RuntimeError(
            f"Checkpoint is unexpectedly small ({path.stat().st_size} bytes): {path}"
        )

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # compatibility with older PyTorch versions
        return torch.load(path, map_location="cpu")


def load_model(device: torch.device) -> torch.nn.Module:
    obj = _load_checkpoint(CHECKPOINT)
    if not isinstance(obj, dict) or "model_state_dict" not in obj:
        raise ValueError(
            "Expected the submitted SemiconRestore-Freq v2 checkpoint dictionary "
            "with a 'model_state_dict' entry."
        )

    config = obj.get("model_config", {})
    model = SemiconRestoreFreqV2(**config)
    model.load_state_dict(obj["model_state_dict"], strict=True)
    model.eval()
    model.to(device)
    return model


def choose_device() -> torch.device:
    requested = os.environ.get("KLA_DEVICE", "").strip()
    if requested:
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("KLA_DEVICE requests CUDA but CUDA is unavailable")
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def choose_batch_size(device: torch.device) -> int:
    override = os.environ.get("KLA_BATCH_SIZE", "").strip()
    if override:
        value = int(override)
        if value < 1:
            raise ValueError("KLA_BATCH_SIZE must be >= 1")
        return value

    if device.type != "cuda":
        return 1

    props = torch.cuda.get_device_properties(device)
    gib = props.total_memory / (1024 ** 3)
    # H100-class memory comfortably supports a larger batch for these 128x128
    # grayscale inputs.  Smaller GPUs automatically use conservative batches.
    if gib >= 60:
        return 64
    if gib >= 30:
        return 32
    if gib >= 14:
        return 16
    if gib >= 7:
        return 8
    return 4


def load_input(path: Path) -> np.ndarray:
    arr = np.load(path, allow_pickle=False).astype(np.float32, copy=False)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Expected a grayscale 2D .npy array: {path} has shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"Input contains NaN or Inf values: {path}")
    # Deliberately do NOT clip input values.  The trained model uses the raw
    # degraded range as part of its degradation-aware feature representation.
    return np.ascontiguousarray(arr)


def chunks(items: list[Path], size: int) -> Iterable[list[Path]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def restore_batch(
    model: torch.nn.Module,
    arrays: list[np.ndarray],
    device: torch.device,
    use_amp: bool,
) -> np.ndarray:
    shapes = {a.shape for a in arrays}
    if len(shapes) != 1:
        raise ValueError(f"A batch contains mixed input shapes: {sorted(shapes)}")

    batch_np = np.stack(arrays, axis=0)[:, None, :, :]
    x = torch.from_numpy(batch_np)
    if device.type == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)

    with torch.inference_mode():
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            y = model(x)

        # The model already clamps to [0, 1].  The explicit sanitation below is
        # an additional submission-contract guard against NaN/Inf or numerical
        # overflow on a different GPU/software stack.
        y = torch.nan_to_num(y.float(), nan=0.0, posinf=1.0, neginf=0.0)
        y = y.clamp_(0.0, 1.0)

    return y[:, 0].cpu().numpy().astype(np.float32, copy=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Restore all semiconductor .npy images in a directory."
    )
    p.add_argument("input_dir", type=Path)
    p.add_argument("output_dir", type=Path)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")

    files = sorted(input_dir.glob("*.npy"))
    if not files:
        raise RuntimeError(f"No .npy files found in: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device()
    use_amp = device.type == "cuda"
    batch_size = choose_batch_size(device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    model = load_model(device)

    device_name = (
        torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    )
    print(f"Device: {device_name}")
    print(f"Images: {len(files)}")
    print(f"Batch size: {batch_size}")
    print(f"AMP FP16: {use_amp}")

    start = time.perf_counter()
    done = 0

    # Dataset images are expected to share a shape.  If an unusual batch has
    # mixed shapes, process those entries individually rather than failing the
    # whole submission.
    for paths in chunks(files, batch_size):
        arrays = [load_input(p) for p in paths]
        if len({a.shape for a in arrays}) == 1:
            preds = restore_batch(model, arrays, device, use_amp)
            pairs = zip(paths, arrays, preds)
        else:
            individual = []
            for path, arr in zip(paths, arrays):
                pred = restore_batch(model, [arr], device, use_amp)[0]
                individual.append((path, arr, pred))
            pairs = individual

        for path, arr, pred in pairs:
            expected = (arr.shape[0] * 2, arr.shape[1] * 2)
            if pred.shape != expected:
                raise RuntimeError(
                    f"Incorrect output resolution for {path.name}: "
                    f"got {pred.shape}, expected {expected}"
                )
            if not np.isfinite(pred).all():
                raise RuntimeError(f"Non-finite output generated for {path.name}")
            if float(pred.min()) < 0.0 or float(pred.max()) > 1.0:
                raise RuntimeError(f"Output outside [0,1] for {path.name}")

            np.save(output_dir / path.name, pred, allow_pickle=False)
            done += 1

        if done % 500 == 0 or done == len(files):
            print(f"Restored {done}/{len(files)}")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    print(f"Completed {done} images in {elapsed:.2f} s")
    if elapsed > 0:
        print(f"End-to-end throughput: {done / elapsed:.2f} images/s")
    print(f"Output directory: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
