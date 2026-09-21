import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):  # t: (B,) in [0,1]
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device).float() / half
        )
        args = t[:, None].float() * freqs[None, :] * 1000.0
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb


class ResBlock1D(nn.Module):
    def __init__(self, channels, cond_dim, dropout=0.05):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, channels), nn.SiLU(),
            nn.Conv1d(channels, channels, 3, padding=1),
        )
        self.cond_proj = nn.Linear(cond_dim, channels)
        self.dropout = nn.Dropout(dropout)
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, channels), nn.SiLU(),
            nn.Conv1d(channels, channels, 3, padding=1),
        )

    def forward(self, x, cond, dropout_mask=None):
        h = self.block1(x)
        h = h + self.cond_proj(cond)[:, :, None]
        # shared-dropout hook: `dropout_mask`, if given, is reused for v_theta and v_theta-
        if dropout_mask is None:
            h = self.dropout(h)
        else:
            h = h * dropout_mask / (1 - self.dropout.p)
        h = self.block2(h)
        return x + h


class Downsample(nn.Module):
    def __init__(self, c): super().__init__(); self.op = nn.Conv1d(c, c, 4, stride=2, padding=1)
    def forward(self, x): return self.op(x)


class Upsample(nn.Module):
    def __init__(self, c): super().__init__(); self.op = nn.ConvTranspose1d(c, c, 4, stride=2, padding=1)
    def forward(self, x): return self.op(x)


class FlowMatchingDecoder(nn.Module):
    """v_theta(t, x_t, mu, spk) -> velocity field, same shape as x_t (B, n_mels, T)."""

    def __init__(self, n_mels=80, base_channels=128, cond_dim=256,
                 n_res_per_level=2, n_levels=3, n_spk=0, spk_emb_dim=64, dropout=0.05):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(cond_dim), nn.Linear(cond_dim, cond_dim), nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.use_spk = n_spk > 0
        if self.use_spk:
            self.spk_emb = nn.Embedding(n_spk, spk_emb_dim)
            self.cond_proj = nn.Linear(cond_dim + spk_emb_dim, cond_dim)

        in_ch = n_mels * 2   # concat(x_t, mu) channel-wise, as in Matcha-TTS
        self.in_conv = nn.Conv1d(in_ch, base_channels, 3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        ch = base_channels
        for lvl in range(n_levels):
            level_blocks = nn.ModuleList(
                [ResBlock1D(ch, cond_dim, dropout) for _ in range(n_res_per_level)]
            )
            self.down_blocks.append(level_blocks)
            self.downsamples.append(Downsample(ch) if lvl < n_levels - 1 else nn.Identity())

        self.mid_blocks = nn.ModuleList([ResBlock1D(ch, cond_dim, dropout) for _ in range(2)])

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for lvl in range(n_levels):
            level_blocks = nn.ModuleList(
                [ResBlock1D(ch, cond_dim, dropout) for _ in range(n_res_per_level)]
            )
            self.up_blocks.append(level_blocks)
            self.upsamples.append(Upsample(ch) if lvl < n_levels - 1 else nn.Identity())

        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv1d(ch, n_mels, 3, padding=1),
        )

    def forward(self, t, x_t, mu, spk_ids=None, shared_dropout_masks=None):
        """
        t: (B,) float in [0,1]
        x_t, mu: (B, n_mels, T)
        shared_dropout_masks: optional list of pre-sampled dropout masks so that
            v_theta and the stop-grad teacher v_theta- use IDENTICAL dropout noise
            (paper §3.2, "Shared Dropout").
        """
        cond = self.time_mlp(t)
        if self.use_spk and spk_ids is not None:
            cond = self.cond_proj(torch.cat([cond, self.spk_emb(spk_ids)], dim=-1))

        h = torch.cat([x_t, mu], dim=1)
        h = self.in_conv(h)

        mask_iter = iter(shared_dropout_masks) if shared_dropout_masks else None
        skips = []
        for level_blocks, down in zip(self.down_blocks, self.downsamples):
            for block in level_blocks:
                m = next(mask_iter) if mask_iter else None
                h = block(h, cond, dropout_mask=m)
            skips.append(h)
            h = down(h)

        for block in self.mid_blocks:
            m = next(mask_iter) if mask_iter else None
            h = block(h, cond, dropout_mask=m)

        for level_blocks, up, skip in zip(self.up_blocks, self.upsamples, reversed(skips)):
            # Handle skip connection size mismatch
            if h.shape[-1] != skip.shape[-1]:
                if h.shape[-1] < skip.shape[-1]:
                    # Pad h to match skip
                    h = torch.nn.functional.pad(h, (0, skip.shape[-1] - h.shape[-1]))
                else:
                    # Pad skip to match h
                    skip = torch.nn.functional.pad(skip, (0, h.shape[-1] - skip.shape[-1]))
            h = h + skip
            for block in level_blocks:
                m = next(mask_iter) if mask_iter else None
                h = block(h, cond, dropout_mask=m)
            h = up(h)

        v = self.out_conv(h)
        # Ensure output matches x_t's time dimension (pad if shorter, truncate if longer)
        target_len = x_t.shape[-1]
        if v.shape[-1] < target_len:
            v = torch.nn.functional.pad(v, (0, target_len - v.shape[-1]))
        else:
            v = v[:, :, :target_len]
        return v

    def sample_shared_dropout_masks(self, x_t, p=0.05, device=None):
        """Pre-sample one dropout mask per ResBlock, reused for teacher and student calls.

        The mask channel dimension must match the U-Net's *internal* feature width
        (base_channels, typically 128), NOT x_t's channel count (n_mels=80).
        Using x_t.shape[1] was the bug that caused the (128 vs 80) RuntimeError.
        """
        n_blocks = sum(len(lv) for lv in self.down_blocks) + len(self.mid_blocks) \
            + sum(len(lv) for lv in self.up_blocks)
        b = x_t.shape[0]
        # in_conv maps (n_mels*2) → base_channels; use its output width for masks
        internal_channels = self.in_conv.out_channels
        masks = [
            (torch.rand(b, internal_channels, 1,
                        device=device or x_t.device) > p).float()
            for _ in range(n_blocks)
        ]
        return masks
