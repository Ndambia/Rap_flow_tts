import torch
import torch.nn as nn
import numpy as np
from numba import njit, prange


class DurationPredictor(nn.Module):
    def __init__(self, dim, kernel_size=3, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2),
            nn.ReLU(), nn.LayerNorm(dim) if False else nn.Identity(),
            nn.Dropout(dropout),
            nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(dim, 1, kernel_size=1),
        )

    def forward(self, x_bct):
        return self.net(x_bct).squeeze(1)   # (B, T_phon) log-duration


@njit(parallel=False)
def _mas(log_p, mask):
    """Monotonic Alignment Search (Glow-TTS), numba-accelerated."""
    T_mel, T_txt = log_p.shape
    Q = np.full((T_mel, T_txt), -np.inf, dtype=np.float64)
    for j in range(T_txt):
        Q[0, j] = log_p[0, j] if j == 0 else -np.inf
    for i in range(1, T_mel):
        for j in range(T_txt):
            if mask[i, j] == 0:
                continue
            prev = Q[i - 1, j]
            if j > 0:
                prev = max(prev, Q[i - 1, j - 1])
            Q[i, j] = prev + log_p[i, j]

    path = np.zeros((T_mel, T_txt), dtype=np.float32)
    j = T_txt - 1
    for i in range(T_mel - 1, -1, -1):
        path[i, j] = 1.0
        if j > 0 and (i == 0 or Q[i - 1, j - 1] >= Q[i - 1, j]):
            j -= 1
    return path


def monotonic_alignment_search(log_prior, phon_lens, mel_lens):
    """log_prior: (B, T_mel, T_txt) -> hard alignment path (B, T_mel, T_txt)."""
    B, T_mel, T_txt = log_prior.shape
    paths = torch.zeros_like(log_prior)
    for b in range(B):
        tm, tt = mel_lens[b].item(), phon_lens[b].item()
        mask = np.ones((tm, tt), dtype=np.uint8)
        p = _mas(log_prior[b, :tm, :tt].detach().cpu().double().numpy(), mask)
        paths[b, :tm, :tt] = torch.from_numpy(p).to(log_prior.device)
    return paths


def expand_by_duration(mu_phoneme, durations):
    """mu_phoneme: (B, C, T_txt), durations: (B, T_txt) int -> (B, C, T_mel)."""
    B, C, T_txt = mu_phoneme.shape
    out = []
    for b in range(B):
        reps = durations[b].clamp(min=0).long()
        out.append(torch.repeat_interleave(mu_phoneme[b], reps, dim=1))
    max_len = max(o.shape[1] for o in out)
    padded = mu_phoneme.new_zeros(B, C, max_len)
    for b, o in enumerate(out):
        padded[b, :, : o.shape[1]] = o
    return padded
