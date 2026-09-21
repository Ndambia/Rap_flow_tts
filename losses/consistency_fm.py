import torch
from .pseudo_huber import pseudo_huber_distance


def sample_xt(x0, x1, t):
    """Linear interpolation: x_t = t*x1 + (1-t)*x0.  t: (B,1,1)."""
    return t * x1 + (1 - t) * x0


def segment_index(t, n_segments):
    """i such that t in [i/S, (i+1)/S)."""
    return torch.clamp((t * n_segments).floor().long(), max=n_segments - 1)


def _align_len(a, b):
    """Trim both tensors along dim-2 to their shared minimum length.

    Stride-2 Conv/ConvTranspose layers don't always round-trip the time
    dimension exactly (e.g. 769 -> 384 -> 768), so v_pred can arrive 1-3
    frames shorter than x_t.  This helper makes every downstream operation
    safe without touching the model weights.
    """
    L = min(a.shape[-1], b.shape[-1])
    return a[..., :L], b[..., :L]


def f_theta(v_pred, t, x_t, n_segments, seg_i):
    """f^i_theta(t, x_t, mu) = x_t + ((i+1)/S - t) * v^i_theta(t, x_t, mu).  Eq.(4)."""
    seg_end = (seg_i.float() + 1) / n_segments
    coeff = (seg_end - t.squeeze(-1).squeeze(-1))[:, None, None]
    v_pred, x_t = _align_len(v_pred, x_t)   # guard against stride-2 length drift
    return x_t + coeff * v_pred


def straight_flow_target_endpoint(x0, x1, t, n_segments, seg_i):
    """x^i = (i+1)/S * x1 + (1-(i+1)/S) * x0  (stage-1 loss target, §3.1 'in practice')."""
    seg_end = ((seg_i.float() + 1) / n_segments)[:, None, None]
    return seg_end * x1 + (1 - seg_end) * x0


class ConsistencyFMLoss:
    """
    Implements:
      Stage 1: L_sf' = || f^i_theta(t, x_t, mu) - x^i ||^2       (straight flow only)
      Stage 2: L_cfm = L_sf + alpha * L_vc                        (Eq. 3-4, full objective)
        L_sf = || f^i_theta(t, x_t, mu) - f^i_theta-(t+dt, x_{t+dt}, mu) ||^2
        L_vc = || v^i_theta(t, x_t, mu) - v^i_theta-(t+dt, x_{t+dt}, mu) ||^2
    Uses the pseudo-Huber metric instead of raw L2 when `use_huber=True` (§3.2).
    """

    def __init__(self, n_segments=2, alpha=1e-5, use_huber=False, n_mels=80, seq_reduce="mean"):
        self.n_segments = n_segments
        self.alpha = alpha
        self.use_huber = use_huber
        self.data_dim = n_mels  # per-frame channel dim; paper uses this for c in Huber metric

    def _metric(self, a, b):
        if self.use_huber:
            return pseudo_huber_distance(a, b, self.data_dim).mean()
        return ((a - b) ** 2).mean()

    def stage1_loss(self, decoder, x0, x1, mu, mel_mask, delta_t_unused=None, spk_ids=None):
        B = x0.shape[0]
        t = torch.rand(B, 1, 1, device=x0.device) * (1 - 1e-4)
        seg_i = segment_index(t.squeeze(-1).squeeze(-1), self.n_segments)
        x_t = sample_xt(x0, x1, t)

        v_pred = decoder(t.squeeze(-1).squeeze(-1), x_t, mu, spk_ids=spk_ids)
        # Trim all tensors to the shortest time dim (guards against stride-2 drift)
        L = min(x_t.shape[-1], v_pred.shape[-1])
        x_t_L      = x_t[..., :L]
        v_pred_L   = v_pred[..., :L]
        x0_L       = x0[..., :L]
        x1_L       = x1[..., :L]
        mel_mask_L = mel_mask[..., :L]

        f_pred   = f_theta(v_pred_L, t, x_t_L, self.n_segments, seg_i)
        x_target = straight_flow_target_endpoint(x0_L, x1_L, t, self.n_segments, seg_i)

        loss = self._metric(f_pred * mel_mask_L, x_target * mel_mask_L)
        return loss

    def stage2_loss(self, decoder, teacher_decoder, x0, x1, mu, mel_mask, delta_t,
                     spk_ids=None, shared_dropout=True):
        B = x0.shape[0]
        t = torch.rand(B, 1, 1, device=x0.device) * (1 - delta_t - 1e-4)
        seg_i = segment_index(t.squeeze(-1).squeeze(-1), self.n_segments)
        x_t = sample_xt(x0, x1, t)

        t_next = t + delta_t
        # keep t_next inside the SAME segment as t, else clip to segment boundary
        seg_end = (seg_i.float() + 1) / self.n_segments
        t_next_clipped = torch.minimum(t_next.squeeze(-1).squeeze(-1), seg_end - 1e-6)[:, None, None]
        x_t_next = sample_xt(x0, x1, t_next_clipped)

        dropout_masks = decoder.sample_shared_dropout_masks(x_t) if shared_dropout else None

        v_student = decoder(t.squeeze(-1).squeeze(-1), x_t, mu, spk_ids=spk_ids,
                             shared_dropout_masks=dropout_masks)
        with torch.no_grad():
            v_teacher = teacher_decoder(
                t_next_clipped.squeeze(-1).squeeze(-1), x_t_next, mu, spk_ids=spk_ids,
                shared_dropout_masks=dropout_masks,
            )

        # Trim everything to the shortest time dim across all tensors
        L = min(x_t.shape[-1], x_t_next.shape[-1],
                v_student.shape[-1], v_teacher.shape[-1])
        x_t_L        = x_t[..., :L]
        x_t_next_L   = x_t_next[..., :L]
        v_student_L  = v_student[..., :L]
        v_teacher_L  = v_teacher[..., :L]
        mel_mask_L   = mel_mask[..., :L]

        f_student = f_theta(v_student_L, t,             x_t_L,      self.n_segments, seg_i)
        f_teacher = f_theta(v_teacher_L, t_next_clipped, x_t_next_L, self.n_segments, seg_i)

        l_sf = self._metric(f_student * mel_mask_L, (f_teacher * mel_mask_L).detach())
        l_vc = self._metric(v_student_L * mel_mask_L, (v_teacher_L * mel_mask_L).detach())

        loss = l_sf + self.alpha * l_vc
        return loss, l_sf.detach(), l_vc.detach()
