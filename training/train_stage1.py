import torch
from torch.utils.data import DataLoader
from model.rapflow_tts import RapFlowTTS
from data.dataset import TTSDataset, collate_fn
from losses.alignment_losses import prior_log_likelihood, duration_loss, prior_loss
from losses.consistency_fm import ConsistencyFMLoss
from model.aligner import monotonic_alignment_search, expand_by_duration


def train_stage1(cfg):
    device = "cuda"
    model = RapFlowTTS(cfg.n_vocab, n_mels=cfg.n_mels, n_spk=cfg.n_spk).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    cfm_loss = ConsistencyFMLoss(n_segments=cfg.n_segments, use_huber=False, n_mels=cfg.n_mels)

    ds = TTSDataset(cfg.train_filelist, multi_speaker=cfg.n_spk > 0)
    dl = DataLoader(ds, batch_size=16, shuffle=True, collate_fn=collate_fn, num_workers=8)

    for epoch in range(cfg.n_epochs_stage1):
        for batch in dl:
            mel = batch["mel"].to(device)
            phon = batch["phonemes"].to(device)
            phon_lens = batch["phon_lens"].to(device)
            mel_lens = batch["mel_lens"].to(device)
            spk = batch["spk_ids"].to(device)

            h, mu_phoneme, log_dur_pred, txt_mask = model.encode(phon, phon_lens)

            log_p_grid = prior_log_likelihood(mel, mu_phoneme, txt_mask)
            with torch.no_grad():
                align_path = monotonic_alignment_search(log_p_grid, phon_lens, mel_lens)
            durations = align_path.sum(1)                       # (B, T_txt)
            mu_expanded = expand_by_duration(mu_phoneme, durations)
            # Ensure mu_expanded matches mel length exactly (pad if too short, truncate if too long)
            if mu_expanded.shape[-1] < mel.shape[-1]:
                padding = mel.shape[-1] - mu_expanded.shape[-1]
                mu_expanded = torch.nn.functional.pad(mu_expanded, (0, padding))
            else:
                mu_expanded = mu_expanded[:, :, : mel.shape[-1]]

            l_dur = duration_loss(log_dur_pred, durations, phon_lens)
            l_prior = prior_loss(mel, mu_expanded, mel_lens)

            mel_mask = (torch.arange(mel.shape[-1], device=device)[None, None, :]
                        < mel_lens[:, None, None]).float()
            x1 = mel
            x0 = torch.randn_like(mel)
            l_sf = cfm_loss.stage1_loss(model.decoder, x0, x1, mu_expanded, mel_mask, spk_ids=spk)

            loss = l_sf + l_dur + l_prior
            opt.zero_grad(); loss.backward(); opt.step()

        torch.save(model.state_dict(), f"{cfg.ckpt_dir}/stage1_epoch{epoch}.pt")
    return model
