import torch
from torch.utils.data import DataLoader
from data.dataset import TTSDataset, collate_fn
from losses.consistency_fm import ConsistencyFMLoss
from model.aligner import monotonic_alignment_search, expand_by_duration
from losses.alignment_losses import prior_log_likelihood
from .trainer_utils import EMA
from schedules.delta_scheduling import make_delta_schedule


def train_stage2(cfg, model):
    device = "cuda"
    # Encoder Freeze (§3.2): freeze text_encoder + duration_predictor for training efficiency
    # and to stabilize consistency training by keeping the condition mu frozen.
    for p in model.text_encoder.parameters():
        p.requires_grad_(False)
    for p in model.duration_predictor.parameters():
        p.requires_grad_(False)

    opt = torch.optim.Adam(model.decoder.parameters(), lr=1e-4)
    ema_teacher = EMA(model.decoder, decay=0.999)
    cfm_loss = ConsistencyFMLoss(n_segments=cfg.n_segments, alpha=1e-5,
                                  use_huber=cfg.use_huber, n_mels=cfg.n_mels)
    get_delta = make_delta_schedule(0.1, 0.001, n_bins=8, n_epochs=cfg.n_epochs_stage2)

    ds = TTSDataset(cfg.train_filelist, multi_speaker=cfg.n_spk > 0)
    dl = DataLoader(ds, batch_size=16, shuffle=True, collate_fn=collate_fn, num_workers=8)

    for epoch in range(cfg.n_epochs_stage2):
        delta_t = get_delta(epoch) if cfg.use_delta_schedule else 0.02
        for batch in dl:
            mel = batch["mel"].to(device)
            phon = batch["phonemes"].to(device)
            phon_lens = batch["phon_lens"].to(device)
            mel_lens = batch["mel_lens"].to(device)
            spk = batch["spk_ids"].to(device)

            with torch.no_grad():
                _, mu_phoneme, _, txt_mask = model.encode(phon, phon_lens)
                log_p_grid = prior_log_likelihood(mel, mu_phoneme, txt_mask)
                align_path = monotonic_alignment_search(log_p_grid, phon_lens, mel_lens)
                durations = align_path.sum(1)
                mu_expanded = expand_by_duration(mu_phoneme, durations)[:, :, : mel.shape[-1]]

            mel_mask = (torch.arange(mel.shape[-1], device=device)[None, None, :]
                        < mel_lens[:, None, None]).float()
            x1 = mel
            x0 = torch.randn_like(mel)

            loss, l_sf, l_vc = cfm_loss.stage2_loss(
                model.decoder, ema_teacher.get(), x0, x1, mu_expanded, mel_mask,
                delta_t=delta_t, spk_ids=spk, shared_dropout=cfg.use_shared_dropout,
            )

            opt.zero_grad(); loss.backward(); opt.step()
            ema_teacher.update(model.decoder)

        torch.save(model.state_dict(), f"{cfg.ckpt_dir}/stage2_epoch{epoch}.pt")
    return model, ema_teacher
