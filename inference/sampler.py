import torch


@torch.no_grad()
def euler_sample(decoder, mu, n_steps=2, spk_ids=None, temperature=1.0):
    """Solve dx/dt = v_theta(t, x_t, mu) from t=0 (noise) to t=1 (data) with an Euler solver.
    mu: (B, n_mels, T) expanded prior. Returns synthesized mel (B, n_mels, T).
    """
    device = mu.device
    x = torch.randn_like(mu) * temperature
    dt = 1.0 / n_steps
    t = torch.zeros(mu.shape[0], device=device)
    for _ in range(n_steps):
        v = decoder(t, x, mu, spk_ids=spk_ids)
        x = x + dt * v
        t = t + dt
    return x
