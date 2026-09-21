"""
apply_fix.py  —  Run this script once in a notebook cell to permanently fix the
stride-2 tensor length mismatch WITHOUT restarting the kernel.

Usage (paste into a notebook cell and run):
    exec(open('apply_fix.py').read())
"""
import sys
import importlib
import types

# ─────────────────────────────────────────────────────────────────────────────
# 1. Write the corrected source file to disk
# ─────────────────────────────────────────────────────────────────────────────
FIXED_SOURCE = '''import torch
from .pseudo_huber import pseudo_huber_distance


def sample_xt(x0, x1, t):
    """Linear interpolation: x_t = t*x1 + (1-t)*x0.  t: (B,1,1)."""
    return t * x1 + (1 - t) * x0


def segment_index(t, n_segments):
    """i such that t in [i/S, (i+1)/S)."""
    return torch.clamp((t * n_segments).floor().long(), max=n_segments - 1)


def _align_len(a, b):
    """Trim both tensors along dim-2 to their shared minimum length.

    Stride-2 Conv/ConvTranspose layers don\'t always round-trip the time
    dimension exactly (e.g. 387 -> 193 -> 384), so v_pred can arrive 1-3
    frames shorter than x_t.  This helper makes every downstream operation
    safe without touching the model weights.
    """
    L = min(a.shape[-1], b.shape[-1])
    return a[..., :L], b[..., :L]


def f_theta(v_pred, t, x_t, n_segments, seg_i):
    """f^i_theta(t, x_t, mu) = x_t + ((i+1)/S - t) * v^i_theta.  Eq.(4)."""
    seg_end = (seg_i.float() + 1) / n_segments
    coeff = (seg_end - t.squeeze(-1).squeeze(-1))[:, None, None]
    v_pred, x_t = _align_len(v_pred, x_t)  # guard against stride-2 drift
    return x_t + coeff * v_pred


def straight_flow_target_endpoint(x0, x1, t, n_segments, seg_i):
    """x^i = (i+1)/S * x1 + (1-(i+1)/S) * x0  (stage-1 target, §3.1)."""
    seg_end = ((seg_i.float() + 1) / n_segments)[:, None, None]
    return seg_end * x1 + (1 - seg_end) * x0


class ConsistencyFMLoss:
    """
    Implements:
      Stage 1: L_sf\' = || f^i_theta(t, x_t, mu) - x^i ||^2
      Stage 2: L_cfm = L_sf + alpha * L_vc   (Eq. 3-4, full objective)
        L_sf = || f^i_theta(t, x_t, mu) - f^i_theta-(t+dt, x_{t+dt}, mu) ||^2
        L_vc = || v^i_theta(t, x_t, mu) - v^i_theta-(t+dt, x_{t+dt}, mu) ||^2
    Uses the pseudo-Huber metric instead of raw L2 when use_huber=True (§3.2).
    """

    def __init__(self, n_segments=2, alpha=1e-5, use_huber=False,
                 n_mels=80, seq_reduce="mean"):
        self.n_segments = n_segments
        self.alpha = alpha
        self.use_huber = use_huber
        self.data_dim = n_mels

    def _metric(self, a, b):
        if self.use_huber:
            return pseudo_huber_distance(a, b, self.data_dim).mean()
        return ((a - b) ** 2).mean()

    def stage1_loss(self, decoder, x0, x1, mu, mel_mask,
                    delta_t_unused=None, spk_ids=None):
        B = x0.shape[0]
        t = torch.rand(B, 1, 1, device=x0.device) * (1 - 1e-4)
        seg_i = segment_index(t.squeeze(-1).squeeze(-1), self.n_segments)
        x_t = sample_xt(x0, x1, t)

        v_pred = decoder(t.squeeze(-1).squeeze(-1), x_t, mu, spk_ids=spk_ids)

        # Trim all frame-axis tensors to the shortest length (stride-2 drift fix)
        L = min(x_t.shape[-1], v_pred.shape[-1])
        x_t_L      = x_t[..., :L]
        v_pred_L   = v_pred[..., :L]
        x0_L       = x0[..., :L]
        x1_L       = x1[..., :L]
        mel_mask_L = mel_mask[..., :L]

        f_pred   = f_theta(v_pred_L, t, x_t_L, self.n_segments, seg_i)
        x_target = straight_flow_target_endpoint(
            x0_L, x1_L, t, self.n_segments, seg_i)

        loss = self._metric(f_pred * mel_mask_L, x_target * mel_mask_L)
        return loss

    def stage2_loss(self, decoder, teacher_decoder, x0, x1, mu, mel_mask,
                    delta_t, spk_ids=None, shared_dropout=True):
        B = x0.shape[0]
        t = torch.rand(B, 1, 1, device=x0.device) * (1 - delta_t - 1e-4)
        seg_i = segment_index(t.squeeze(-1).squeeze(-1), self.n_segments)
        x_t = sample_xt(x0, x1, t)

        t_next = t + delta_t
        seg_end = (seg_i.float() + 1) / self.n_segments
        t_next_clipped = torch.minimum(
            t_next.squeeze(-1).squeeze(-1), seg_end - 1e-6)[:, None, None]
        x_t_next = sample_xt(x0, x1, t_next_clipped)

        dropout_masks = (decoder.sample_shared_dropout_masks(x_t)
                         if shared_dropout else None)

        v_student = decoder(
            t.squeeze(-1).squeeze(-1), x_t, mu,
            spk_ids=spk_ids, shared_dropout_masks=dropout_masks)
        with torch.no_grad():
            v_teacher = teacher_decoder(
                t_next_clipped.squeeze(-1).squeeze(-1), x_t_next, mu,
                spk_ids=spk_ids, shared_dropout_masks=dropout_masks)

        # Trim to shortest across all frame-axis tensors
        L = min(x_t.shape[-1], x_t_next.shape[-1],
                v_student.shape[-1], v_teacher.shape[-1])
        x_t_L       = x_t[..., :L]
        x_t_next_L  = x_t_next[..., :L]
        v_student_L = v_student[..., :L]
        v_teacher_L = v_teacher[..., :L]
        mel_mask_L  = mel_mask[..., :L]

        f_student = f_theta(
            v_student_L, t, x_t_L, self.n_segments, seg_i)
        f_teacher = f_theta(
            v_teacher_L, t_next_clipped, x_t_next_L, self.n_segments, seg_i)

        l_sf = self._metric(
            f_student * mel_mask_L, (f_teacher * mel_mask_L).detach())
        l_vc = self._metric(
            v_student_L * mel_mask_L, (v_teacher_L * mel_mask_L).detach())

        loss = l_sf + self.alpha * l_vc
        return loss, l_sf.detach(), l_vc.detach()
'''

# Write corrected source to disk
import os
from pathlib import Path

losses_dir = Path(__file__).parent / "losses" if "__file__" in dir() else Path(".") / "losses"
target = losses_dir / "consistency_fm.py"

with open(target, "w", encoding="utf-8") as fh:
    fh.write(FIXED_SOURCE)

print(f"✅ Wrote fixed source to {target}")

# Delete stale __pycache__ for this module
pycache = losses_dir / "__pycache__"
if pycache.exists():
    for f in pycache.glob("consistency_fm*.pyc"):
        f.unlink()
        print(f"   Deleted cached bytecode: {f.name}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Force-reload the module in the live Python session
# ─────────────────────────────────────────────────────────────────────────────
mod_name = "losses.consistency_fm"
if mod_name in sys.modules:
    del sys.modules[mod_name]

import losses.consistency_fm as cfm_mod  # fresh import from new source

# ─────────────────────────────────────────────────────────────────────────────
# 3. Monkey-patch every live ConsistencyFMLoss instance in the namespace
# ─────────────────────────────────────────────────────────────────────────────
import gc

fixed_class = cfm_mod.ConsistencyFMLoss
patched = 0

for obj in gc.get_objects():
    if type(obj).__name__ == "ConsistencyFMLoss":
        # Swap the __class__ pointer so the live instance uses the new methods
        try:
            obj.__class__ = fixed_class
            patched += 1
        except Exception:
            pass

# Also patch any module-level references in all loaded modules
for name, mod in list(sys.modules.items()):
    if mod is None:
        continue
    for attr in list(vars(mod).keys()):
        try:
            val = getattr(mod, attr)
            if val is cfm_mod.f_theta or (
                callable(val) and getattr(val, "__name__", "") == "f_theta"
                and "coeff * v_pred" in (getattr(val, "__doc__", "") or "")
            ):
                setattr(mod, attr, cfm_mod.f_theta)
        except Exception:
            pass

# ─────────────────────────────────────────────────────────────────────────────
# 4. Verify the fix is live
# ─────────────────────────────────────────────────────────────────────────────
import inspect
src = inspect.getsource(cfm_mod.f_theta)
assert "_align_len" in src, "❌ Fix did NOT take effect in f_theta!"

src2 = inspect.getsource(cfm_mod.ConsistencyFMLoss.stage1_loss)
assert "v_pred.shape[-1]" in src2, "❌ Fix did NOT take effect in stage1_loss!"

print(f"✅ Module reloaded — _align_len is active in f_theta")
print(f"✅ stage1_loss length-trim is active")
print(f"✅ Monkey-patched {patched} live ConsistencyFMLoss instance(s)")
print()
print("👉 NOW: re-create cfm_loss to pick up the fixed class:")
print("   cfm_loss = ConsistencyFMLoss(n_segments=N_SEGMENTS, use_huber=False, n_mels=N_MELS)")
print("   from losses.consistency_fm import ConsistencyFMLoss")
