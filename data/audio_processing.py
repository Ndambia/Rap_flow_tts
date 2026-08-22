import torch
import torchaudio

MEL_CONFIG = dict(
    sample_rate=22050,
    n_fft=1024,
    win_length=1024,
    hop_length=256,
    n_mels=80,
    f_min=0,
    f_max=8000,
)

_mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=MEL_CONFIG["sample_rate"],
    n_fft=MEL_CONFIG["n_fft"],
    win_length=MEL_CONFIG["win_length"],
    hop_length=MEL_CONFIG["hop_length"],
    f_min=MEL_CONFIG["f_min"],
    f_max=MEL_CONFIG["f_max"],
    n_mels=MEL_CONFIG["n_mels"],
    power=1.0,
    norm=None,
    mel_scale="slaney",
)


def wav_to_mel(wav: torch.Tensor) -> torch.Tensor:
    """wav: (1, T) float32 in [-1, 1] -> log-mel (n_mels, frames)."""
    mel = _mel_transform(wav)
    log_mel = torch.log(torch.clamp(mel, min=1e-5))
    return log_mel.squeeze(0)
