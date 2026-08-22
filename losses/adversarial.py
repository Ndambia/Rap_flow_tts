import torch


def lsgan_losses(disc, x_hat_i, x_gt_i):
    """LSGAN discriminator/generator losses + feature matching, on segment endpoints."""
    d_fake, feats_fake = disc(x_hat_i)
    d_real, feats_real = disc(x_gt_i.detach())

    d_loss = (d_fake ** 2).mean() + ((1 - d_real) ** 2).mean()

    d_fake_for_g, feats_fake_for_g = disc(x_hat_i)
    g_adv_loss = ((1 - d_fake_for_g) ** 2).mean()

    fm_loss = sum(
        (f_fake - f_real.detach()).abs().mean()
        for f_fake, f_real in zip(feats_fake_for_g, feats_real)
    ) / len(feats_real)

    return d_loss, g_adv_loss, fm_loss
