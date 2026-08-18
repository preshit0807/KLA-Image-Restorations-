"""
SemiconRestore-Freq v2
======================
A strict extension of the user's already-trained SemiconRestoreDA.

Key invariant:
    with the zero-initialized detail adapter, this model produces EXACTLY the
    same output as SemiconRestoreDA loaded from the same checkpoint.

Only after fine-tuning can the new Haar feature adapter change the output.
The adapter predicts high-frequency-only corrections (zero LL Haar band),
which constrains it from directly changing each 2x2 block's local average.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_semiconrestore import SemiconRestoreDA, build_informational_channels


def haar_dwt2(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if x.ndim != 4:
        raise ValueError(f"Expected BCHW tensor, got {tuple(x.shape)}")
    if x.shape[-2] % 2 or x.shape[-1] % 2:
        raise ValueError(f"Haar DWT requires even H,W, got {tuple(x.shape[-2:])}")
    a = x[..., 0::2, 0::2]
    b = x[..., 0::2, 1::2]
    c = x[..., 1::2, 0::2]
    d = x[..., 1::2, 1::2]
    ll = (a + b + c + d) * 0.5
    lh = (a - b + c - d) * 0.5
    hl = (a + b - c - d) * 0.5
    hh = (a - b - c + d) * 0.5
    return ll, lh, hl, hh


def haar_idwt2(
    ll: torch.Tensor,
    lh: torch.Tensor,
    hl: torch.Tensor,
    hh: torch.Tensor,
) -> torch.Tensor:
    if not (ll.shape == lh.shape == hl.shape == hh.shape):
        raise ValueError("All Haar bands must have identical shapes")
    a = (ll + lh + hl + hh) * 0.5
    b = (ll - lh + hl - hh) * 0.5
    c = (ll + lh - hl - hh) * 0.5
    d = (ll - lh - hl + hh) * 0.5
    bsz, ch, h, w = ll.shape
    out = ll.new_empty((bsz, ch, h * 2, w * 2))
    out[..., 0::2, 0::2] = a
    out[..., 0::2, 1::2] = b
    out[..., 1::2, 0::2] = c
    out[..., 1::2, 1::2] = d
    return out


class DetailResidualBlock(nn.Module):
    """Cheap depthwise-gated residual block used only inside the new adapter."""
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.in_proj = nn.Conv2d(channels, channels * 2, 1)
        self.dw = nn.Conv2d(channels * 2, channels * 2, 3, padding=1, groups=channels * 2)
        self.out_proj = nn.Conv2d(channels, channels, 1)
        self.scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        y = self.in_proj(y)
        y = self.dw(y)
        a, b = y.chunk(2, dim=1)
        y = a * b
        y = self.out_proj(y)
        return x + self.scale * y


class HaarFeatureDetailAdapter(nn.Module):
    """
    Reads the pretrained model's final HR feature tensor and predicts only
    Haar detail-band corrections (LH/HL/HH). The LL correction is fixed zero.

    The final projection is exactly zero-initialized, so adapter output is
    exactly zero at initialization while its output layer receives gradients
    on the first optimization step.
    """
    def __init__(self, feature_ch: int = 32, hidden: int = 32, blocks: int = 2,
                 max_correction: float = 0.25):
        super().__init__()
        if hidden <= 0 or blocks < 0:
            raise ValueError("Invalid adapter size")
        self.max_correction = float(max_correction)
        self.in_proj = nn.Conv2d(feature_ch * 3, hidden, 1)
        self.blocks = nn.Sequential(*[DetailResidualBlock(hidden) for _ in range(blocks)])
        self.out_proj = nn.Conv2d(hidden, 3, 3, padding=1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, hr_features: torch.Tensor) -> torch.Tensor:
        _, lh, hl, hh = haar_dwt2(hr_features)
        y = self.in_proj(torch.cat([lh, hl, hh], dim=1))
        y = self.blocks(y)
        bands = self.out_proj(y)
        dl_h, dh_l, dh_h = bands.chunk(3, dim=1)
        zero_ll = torch.zeros_like(dl_h)
        correction = haar_idwt2(zero_ll, dl_h, dh_l, dh_h)
        return self.max_correction * torch.tanh(correction)


class SemiconRestoreFreqV2(SemiconRestoreDA):
    """Exact SemiconRestoreDA backbone + zero-init localized detail adapter."""
    def __init__(
        self,
        in_ch: int = 1,
        base_ch: int = 32,
        enc_blocks=(2, 2, 4),
        middle_blocks: int = 4,
        dec_blocks=(2, 2, 2),
        upscale: int = 2,
        z_dim: int = 64,
        asinh_scale: float = 0.2,
        adapter_hidden: int = 32,
        adapter_blocks: int = 2,
        adapter_max_correction: float = 0.25,
    ):
        super().__init__(
            in_ch=in_ch,
            base_ch=base_ch,
            enc_blocks=enc_blocks,
            middle_blocks=middle_blocks,
            dec_blocks=dec_blocks,
            upscale=upscale,
            z_dim=z_dim,
            asinh_scale=asinh_scale,
        )
        self.detail_adapter = HaarFeatureDetailAdapter(
            feature_ch=base_ch,
            hidden=adapter_hidden,
            blocks=adapter_blocks,
            max_correction=adapter_max_correction,
        )
        self.v2_config = {
            "in_ch": in_ch,
            "base_ch": base_ch,
            "enc_blocks": tuple(enc_blocks),
            "middle_blocks": middle_blocks,
            "dec_blocks": tuple(dec_blocks),
            "upscale": upscale,
            "z_dim": z_dim,
            "asinh_scale": asinh_scale,
            "adapter_hidden": adapter_hidden,
            "adapter_blocks": adapter_blocks,
            "adapter_max_correction": adapter_max_correction,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # This section intentionally mirrors model_semiconrestore.SemiconRestoreDA.forward.
        base = F.interpolate(x, scale_factor=self.upscale, mode="bicubic", align_corners=False)
        base = torch.clamp(base, 0, 1)

        info = build_informational_channels(x, asinh_scale=self.asinh_scale)
        z = self.degradation_encoder(info)
        feat = self.intro(info)

        skips = []
        for enc_blocks, down in zip(self.encoders, self.downs):
            for block in enc_blocks:
                feat = block(feat, z)
            skips.append(feat)
            feat = down(feat)

        for block in self.middle:
            feat = block(feat, z)

        for up, fuse, dec_blocks, skip in zip(
            self.ups, self.fuse_convs, self.decoders, reversed(skips)
        ):
            feat = up(feat)
            feat = torch.cat([feat, skip], dim=1)
            feat = fuse(feat)
            for block in dec_blocks:
                feat = block(feat, z)

        # Exact pretrained path.
        feat = self.final_upsample(feat)
        residual = self.tail(feat)

        # New branch. At initialization detail_correction == 0 exactly.
        detail_correction = self.detail_adapter(feat)
        out = torch.clamp(base + residual + detail_correction, 0, 1)
        return out


def _extract_state_dict(obj):
    if isinstance(obj, dict) and "model_state_dict" in obj:
        return obj["model_state_dict"]
    return obj


def load_base_checkpoint(model: SemiconRestoreFreqV2, checkpoint: str | Path) -> Dict[str, list]:
    """Load a plain SemiconRestoreDA checkpoint into the v2 superset safely."""
    checkpoint = Path(checkpoint)
    obj = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = _extract_state_dict(obj)
    result = model.load_state_dict(state, strict=False)
    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)

    illegal_missing = [k for k in missing if not k.startswith("detail_adapter.")]
    if illegal_missing or unexpected:
        raise RuntimeError(
            "Base checkpoint is not compatible with exact SemiconRestoreDA backbone. "
            f"Illegal missing={illegal_missing[:10]}, unexpected={unexpected[:10]}"
        )
    return {"missing": missing, "unexpected": unexpected}


def configure_trainable_stage(model: SemiconRestoreFreqV2, stage: str) -> None:
    """Freeze/unfreeze parameters for controlled fine-tuning."""
    for p in model.parameters():
        p.requires_grad = False

    if stage == "adapter":
        for p in model.detail_adapter.parameters():
            p.requires_grad = True
        return

    if stage == "late":
        modules = [model.detail_adapter, model.final_upsample, model.tail]
        if len(model.decoders):
            modules += [model.decoders[-1], model.fuse_convs[-1], model.ups[-1]]
        for module in modules:
            for p in module.parameters():
                p.requires_grad = True
        return

    if stage == "full":
        for p in model.parameters():
            p.requires_grad = True
        return

    raise ValueError(f"Unknown stage {stage!r}; expected adapter/late/full")


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(
        p.numel() for p in model.parameters()
        if (p.requires_grad or not trainable_only)
    )
