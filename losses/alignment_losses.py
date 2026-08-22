import torch
import torch.nn.functional as F


def prior_log_likelihood(mel, mu_phoneme, mask_txt, log_scale=0.0):
    """Diagonal-Gaussian log p(mel_i | phoneme_j) grid, used for MAS.
    mel: (B, n_mels, T_mel), mu_phoneme: (B, n_mels, T_txt)
    Returns (B, T_mel, T_txt) log-likelihood grid.
    """
    B, C, T_mel = mel.shape
    T_txt = mu_phoneme.shape[-1]
    mel_sq = (mel ** 2).sum(1, keepdim=True).transpose(1, 2)         # (B, T_mel, 1)
    mu_sq = (mu_phoneme ** 2).sum(1, keepdim=True)                    # (B, 1, T_txt)
    cross = torch.einsum("bct,bcs->bts", mel, mu_phoneme)             # (B, T_mel, T_txt)
    log_p = -0.5 * (mel_sq + mu_sq - 2 * cross) - 0.5 * C * (2 * log_scale + 1.837877)
    log_p = log_p.masked_fill(mask_txt[:, None, :], float("-inf"))
    return log_p


def duration_loss(log_dur_pred, durations, phon_lens):
    mask = torch.arange(log_dur_pred.shape[1], device=log_dur_pred.device)[None, :] \
        < phon_lens[:, None]
    log_dur_target = torch.log(durations.float().clamp(min=1))
    loss = F.mse_loss(log_dur_pred[mask], log_dur_target[mask])
    return loss


def prior_loss(mel, mu_expanded, mel_lens):
    mask = torch.arange(mel.shape[-1], device=mel.device)[None, :] < mel_lens[:, None]
    diff2 = (mel - mu_expanded) ** 2
    loss = (diff2.mean(1) * mask).sum() / mask.sum()
    return loss
