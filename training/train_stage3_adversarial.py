import torch
from torch.utils.data import DataLoader
from data.dataset import TTSDataset, collate_fn
from model.discriminator import Conv2dDiscriminator
from losses.consistency_fm import (
    ConsistencyFMLoss, sample_xt, segment_index, f_theta, straight_flow_target_endpoint,
)
from losses.adversarial import lsgan_losses
from model.aligner import monotonic_alignment_search, expand_by_duration
from losses.alignment_losses import prior_log_likelihood


def train_stage3_adversarial(cfg, model, ema_teacher):
    device = "cuda"
    disc = Conv2dDiscriminator().to(device)
    opt_g = torch.optim.Adam(model.decoder.parameters(), lr=1e-4)
    opt_d = torch.optim.Adam(disc.parameters(), lr=1e-4)

    cfm_loss = ConsistencyFMLoss(n_segments=cfg.n_segments, alpha=1e-5,
                                  use_huber=cfg.use_huber, n_mels=cfg.n_mels)
    delta_t = 0.001   # end of delta schedule

    ds = TTSDataset(cfg.train_filelist, multi_speaker=cfg.n_spk > 0)
    dl = DataLoader(ds, batch_size=16, shuffle=True, collate_fn=collate_fn, num_workers=8)
    n_epochs_adv = cfg.n_epochs_adversarial   # 150 (LJSpeech) / 50 (VCTK) per paper §4.1

    for epoch in range(n_epochs_adv):
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
            x1, x0 = mel, torch.randn_like(mel)

            # --- consistency FM loss (student vs EMA teacher) ---
            l_cfm, l_sf, l_vc = cfm_loss.stage2_loss(
                model.decoder, ema_teacher.get(), x0, x1, mu_expanded, mel_mask,
                delta_t=delta_t, spk_ids=spk, shared_dropout=True,
            )

            # --- segment endpoints for adversarial loss (Eq. 5) ---
            B = x0.shape[0]
            t = torch.rand(B, 1, 1, device=device) * (1 - 1e-4)
            seg_i = segment_index(t.squeeze(-1).squeeze(-1), cfg.n_segments)
            x_t = sample_xt(x0, x1, t)
            v_pred = model.decoder(t.squeeze(-1).squeeze(-1), x_t, mu_expanded, spk_ids=spk)
            x_hat_i = f_theta(v_pred, t, x_t, cfg.n_segments, seg_i) * mel_mask
            x_gt_i = straight_flow_target_endpoint(x0, x1, t, cfg.n_segments, seg_i) * mel_mask

            # --- discriminator step ---
            d_loss, g_adv_loss, fm_loss = lsgan_losses(disc, x_hat_i, x_gt_i)
            opt_d.zero_grad(); d_loss.backward(retain_graph=True); opt_d.step()

            # --- generator step: ratio 3 (cfm) : 1 (adv) : 2 (fm) ---
            g_loss = 3 * l_cfm + 1 * g_adv_loss + 2 * fm_loss
            opt_g.zero_grad(); g_loss.backward(); opt_g.step()
            ema_teacher.update(model.decoder)

        torch.save(model.state_dict(), f"{cfg.ckpt_dir}/stage3_epoch{epoch}.pt")
        torch.save(disc.state_dict(), f"{cfg.ckpt_dir}/disc_epoch{epoch}.pt")
    return model
