"""
SemiconRestore-DA: a degradation-aware extension of the NAFNet-style model.

Two additions over model_nafnet.py, both chosen for being cheap, deterministic,
and NON-destructive (they add information, never remove/filter it):

1. MULTI-CHANNEL INFORMATIONAL STEM
   Instead of feeding the network only the raw noisy pixel value, we also
   compute three deterministic auxiliary channels and concatenate them:
     - raw intensity            (unchanged, exact signal)
     - asinh-compressed intensity: asinh(x / s.) behaves ~linearly near
       zero but compresses extreme excursions. Unlike log(x), it's defined
       for the negative values that occur in this dataset (measured range
       ~[-0.28, 2.16]), so it doesn't break on real inputs the way a plain
       log-domain transform would.
     - local morphological gradient: dilate(x) - erode(x) via 3x3 max/min
       pooling. Responds strongly to edges, thin lines, small high-contrast
       structures -- exactly the fine detail (spikes, whiskers, structure
       edges) that plain NAFNet sometimes over-smoothed.
     - local variance: a cheap per-pixel noise-vs-signal indicator (high in
       noisy/textured regions, low in flat regions).
   These are all fixed, non-learnable transforms of the input computed once
   per image at near-zero cost -- they give the network immediate access to
   physically meaningful cues instead of requiring it to rediscover them
   from raw pixels alone.

2. LIGHTWEIGHT DEGRADATION ENCODER + FiLM CONDITIONING
   A small encoder compresses the input into a low-dimensional "degradation
   code" describing how corrupted this particular image is (continuous, not
   a hard Gaussian/speckle/clean classification, since real images contain
   mixtures). Each NAFBlock is modulated by this code via FiLM (feature-wise
   linear modulation: scale + shift applied to normalized features),
   letting the network adapt its processing per-image rather than applying
   identical operations regardless of degradation severity.

Both additions preserve the validated NAFNet backbone (activation-free
blocks, simplified channel attention, U-Net skip connections, residual
learning with zero-initialized output) -- they extend it, rather than
replacing something that already works.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Reused building blocks (same as model_nafnet.py)
# ---------------------------------------------------------------------------

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimplifiedChannelAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        attn = self.conv(self.pool(x))
        return x * attn


# ---------------------------------------------------------------------------
# New: FiLM-conditioned NAFBlock
# ---------------------------------------------------------------------------

class FiLMGenerator(nn.Module):
    """Maps a degradation code z -> per-channel (scale, shift) for one block."""

    def __init__(self, z_dim, channels):
        super().__init__()
        self.net = nn.Linear(z_dim, channels * 2)
        # zero-init: at the start of training, FiLM is the identity transform
        # (scale=1, shift=0) so this doesn't destabilize the already-validated
        # NAFBlock behavior before the network has learned anything useful
        nn.init.zeros_(self.net.weight)
        nn.init.zeros_(self.net.bias)
        self.channels = channels

    def forward(self, z):
        out = self.net(z)  # (B, channels*2)
        gamma, beta = out.chunk(2, dim=-1)
        gamma = gamma.view(-1, self.channels, 1, 1)
        beta = beta.view(-1, self.channels, 1, 1)
        return gamma, beta


class FiLMNAFBlock(nn.Module):
    """
    Same structure as NAFBlock (model_nafnet.py), with FiLM modulation
    applied to the normalized features in each sub-block, conditioned on
    the degradation code z. If z is None, behaves identically to a plain
    NAFBlock (useful for the middle/decoder stages if you want to skip
    conditioning there).
    """

    def __init__(self, channels, z_dim=None, expand_ratio=2):
        super().__init__()
        hidden = channels * expand_ratio
        self.z_dim = z_dim

        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, hidden, 1)
        self.dwconv = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.gate1 = SimpleGate()
        self.sca = SimplifiedChannelAttention(hidden // 2)
        self.conv1_out = nn.Conv2d(hidden // 2, channels, 1)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

        self.norm2 = LayerNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, hidden, 1)
        self.gate2 = SimpleGate()
        self.conv2_out = nn.Conv2d(hidden // 2, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

        if z_dim is not None:
            self.film1 = FiLMGenerator(z_dim, channels)
            self.film2 = FiLMGenerator(z_dim, channels)

    def _apply_film(self, x, film, z):
        if z is None or self.z_dim is None:
            return x
        gamma, beta = film(z)
        return (1 + gamma) * x + beta

    def forward(self, x, z=None):
        residual = x
        y = self.norm1(x)
        y = self._apply_film(y, self.film1, z) if self.z_dim is not None else y
        y = self.conv1(y)
        y = self.dwconv(y)
        y = self.gate1(y)
        y = self.sca(y)
        y = self.conv1_out(y)
        x = residual + y * self.beta

        residual = x
        y = self.norm2(x)
        y = self._apply_film(y, self.film2, z) if self.z_dim is not None else y
        y = self.conv2(y)
        y = self.gate2(y)
        y = self.conv2_out(y)
        x = residual + y * self.gamma

        return x


# ---------------------------------------------------------------------------
# Degradation encoder
# ---------------------------------------------------------------------------

class DegradationEncoder(nn.Module):
    """
    Small CNN that compresses the (multi-channel) input into a continuous
    degradation code z. Deliberately shallow/cheap -- its job is just to
    summarize "how corrupted and in what way", not to do restoration itself.
    """

    def __init__(self, in_ch=4, z_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, 3, stride=2, padding=1, groups=32),  # depthwise
            nn.Conv2d(32, 64, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1, groups=64),  # depthwise
            nn.Conv2d(64, 96, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96, 96),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(96, z_dim),
        )

    def forward(self, x):
        feat = self.net(x)
        z = self.mlp(feat)
        return z


# ---------------------------------------------------------------------------
# Deterministic auxiliary channel construction
# ---------------------------------------------------------------------------

def build_informational_channels(x, asinh_scale=0.2):
    """
    x: (B, 1, H, W) raw noisy input (may contain values outside [0,1]).
    Returns (B, 4, H, W): [raw, asinh-compressed, morphological gradient,
    local variance].
    """
    raw = x

    # asinh compression: ~linear near zero, compresses large excursions,
    # defined for negative inputs (unlike log), matches this dataset's
    # measured value range which goes slightly negative.
    compressed = torch.asinh(x / asinh_scale)
    # rescale roughly back toward a similar numeric range as raw, so it
    # doesn't dominate purely due to differing magnitude
    compressed = compressed * asinh_scale

    # morphological gradient: dilate - erode via 3x3 max/min pooling
    dilated = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
    eroded = -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)
    morph_grad = dilated - eroded

    # local variance via unfold: E[x^2] - E[x]^2 over a 5x5 window
    kernel_size = 5
    pad = kernel_size // 2
    x_mean = F.avg_pool2d(x, kernel_size, stride=1, padding=pad)
    x_sq_mean = F.avg_pool2d(x * x, kernel_size, stride=1, padding=pad)
    local_var = (x_sq_mean - x_mean ** 2).clamp(min=0)

    return torch.cat([raw, compressed, morph_grad, local_var], dim=1)


# ---------------------------------------------------------------------------
# Downsample / Upsample (same as model_nafnet.py)
# ---------------------------------------------------------------------------

class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch * 2, 2, stride=2)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch * 2, 1)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        return self.shuffle(self.conv(x))


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class SemiconRestoreDA(nn.Module):
    def __init__(self, in_ch=1, base_ch=32, enc_blocks=(2, 2, 4), middle_blocks=4,
                 dec_blocks=(2, 2, 2), upscale=2, z_dim=64, asinh_scale=0.2):
        super().__init__()
        self.upscale = upscale
        self.asinh_scale = asinh_scale

        # 4 informational channels (raw, asinh, morph-gradient, local-var) -> stem
        self.intro = nn.Conv2d(4, base_ch, 3, padding=1)

        self.degradation_encoder = DegradationEncoder(in_ch=4, z_dim=z_dim)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        ch = base_ch
        for n in enc_blocks:
            self.encoders.append(nn.ModuleList(
                [FiLMNAFBlock(ch, z_dim=z_dim) for _ in range(n)]
            ))
            self.downs.append(Downsample(ch))
            ch *= 2

        self.middle = nn.ModuleList(
            [FiLMNAFBlock(ch, z_dim=z_dim) for _ in range(middle_blocks)]
        )

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.fuse_convs = nn.ModuleList()
        for n in dec_blocks:
            self.ups.append(Upsample(ch))
            ch //= 2
            self.fuse_convs.append(nn.Conv2d(ch * 2, ch, 1))
            self.decoders.append(nn.ModuleList(
                [FiLMNAFBlock(ch, z_dim=z_dim) for _ in range(n)]
            ))

        final_up = []
        c = ch
        n_up = int(math.log2(upscale))
        for _ in range(max(n_up, 1)):
            final_up += [
                nn.Conv2d(c, c * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        self.final_upsample = nn.Sequential(*final_up)
        self.tail = nn.Conv2d(c, in_ch, 3, padding=1)

        # zero-init tail: training starts from "predict zero correction",
        # i.e. output == naive bicubic upsample (stable starting point)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, x):
        # naive baseline the network only needs to *correct*
        base = F.interpolate(x, scale_factor=self.upscale, mode="bicubic",
                              align_corners=False)
        base = torch.clamp(base, 0, 1)

        # build informational multi-channel stem + degradation code
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

        for up, fuse, dec_blocks, skip in zip(self.ups, self.fuse_convs,
                                               self.decoders, reversed(skips)):
            feat = up(feat)
            feat = torch.cat([feat, skip], dim=1)
            feat = fuse(feat)
            for block in dec_blocks:
                feat = block(feat, z)

        feat = self.final_upsample(feat)
        residual = self.tail(feat)

        out = torch.clamp(base + residual, 0, 1)
        return out


if __name__ == "__main__":
    model = SemiconRestoreDA()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params:,}")

    x1 = torch.randn(2, 1, 64, 64)
    y1 = model(x1)
    print(f"64x64 input -> {tuple(y1.shape)} (expected 128x128)")
    assert y1.shape == (2, 1, 128, 128)

    x2 = torch.randn(1, 1, 128, 128)
    y2 = model(x2)
    print(f"128x128 input -> {tuple(y2.shape)} (expected 256x256)")
    assert y2.shape == (1, 1, 256, 256)

    print("Output range:", y2.min().item(), y2.max().item())

    # zero-init check: output should equal naive upsample at init
    with torch.no_grad():
        base = torch.clamp(F.interpolate(x2, scale_factor=2, mode="bicubic",
                                          align_corners=False), 0, 1)
        diff = (y2 - base).abs().max().item()
    print(f"Max diff from naive upsample at init: {diff:.6f} (should be ~0)")

    print("SEMICONRESTORE-DA MODEL OK")
