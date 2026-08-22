import torch
import torch.nn as nn
from einops import rearrange


class FFTBlock(nn.Module):
    def __init__(self, dim, n_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Conv1d(dim, ffn_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(ffn_dim, dim, kernel_size=3, padding=1),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = residual + self.dropout(attn_out)

        residual = x
        x = self.norm2(x)
        x = rearrange(x, "b t c -> b c t")
        x = self.ffn(x)
        x = rearrange(x, "b c t -> b t c")
        x = residual + self.dropout(x)
        return x


class TextEncoder(nn.Module):
    def __init__(self, n_vocab, dim=192, n_layers=6, n_heads=2, ffn_dim=768,
                 n_mels=80, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(n_vocab, dim)
        nn.init.normal_(self.emb.weight, 0.0, dim ** -0.5)
        self.blocks = nn.ModuleList(
            [FFTBlock(dim, n_heads, ffn_dim, dropout) for _ in range(n_layers)]
        )
        self.proj_mu = nn.Conv1d(dim, n_mels, kernel_size=1)   # -> per-phoneme prior stats

    def forward(self, phonemes, phon_lens):
        mask = torch.arange(phonemes.shape[1], device=phonemes.device)[None, :] \
            >= phon_lens[:, None]                                # True where padding
        x = self.emb(phonemes)
        for block in self.blocks:
            x = block(x, key_padding_mask=mask)
        mu_phoneme = self.proj_mu(rearrange(x, "b t c -> b c t"))  # (B, n_mels, T_phon)
        return x, mu_phoneme, mask
