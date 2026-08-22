import torch
from data.text_processing import text_to_phoneme_ids
from model.aligner import expand_by_duration
from .sampler import euler_sample


@torch.no_grad()
def synthesize(model, text, hifigan, device="cuda", n_steps=2, spk_id=0):
    phon = torch.LongTensor(text_to_phoneme_ids(text))[None].to(device)
    phon_lens = torch.LongTensor([phon.shape[1]]).to(device)
    spk = torch.LongTensor([spk_id]).to(device)

    h, mu_phoneme, log_dur_pred, _ = model.encode(phon, phon_lens)
    durations = torch.exp(log_dur_pred).round().clamp(min=1)
    mu_expanded = expand_by_duration(mu_phoneme, durations)

    mel = euler_sample(model.decoder, mu_expanded, n_steps=n_steps, spk_ids=spk)
    wav = hifigan(mel).squeeze().cpu().numpy()     # frozen, pretrained HiFi-GAN V1
    return wav, mel
