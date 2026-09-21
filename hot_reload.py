"""
hot_reload.py — run this in the first cell of every training step
to force Python to re-read the patched source files.

Usage (copy into a notebook cell, run BEFORE any training cell):
    from hot_reload import reload_fixed_modules
    reload_fixed_modules()
"""
import importlib
import sys


def reload_fixed_modules():
    """
    Force-reload the two modules that were patched to fix the
    stride-2 tensor-length mismatch (RuntimeError at f_theta line 19).

    Call this once per kernel session, right before training starts.
    """
    # 1. Reload the flow decoder (pad output to match x_t length)
    import model.flow_decoder as _fd
    importlib.reload(_fd)

    # 2. Re-import FlowMatchingDecoder so the class uses the reloaded code
    import model.rapflow_tts as _rft
    importlib.reload(_rft)

    # 3. Reload the loss module (_align_len helper + stage1/stage2 length guards)
    import losses.consistency_fm as _cfm
    importlib.reload(_cfm)

    print("✅ hot_reload: flow_decoder, rapflow_tts, consistency_fm reloaded.")
    print("   Tensor-length mismatch fix is now active.")
    return _rft, _cfm


if __name__ == "__main__":
    reload_fixed_modules()
