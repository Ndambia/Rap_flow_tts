"""
Builds a MOS listening-test package: for each system, randomly sample 20 utterances
(matching the paper's protocol: 20 participants, 20 utterances/system, scale 1-5),
and export a manifest + a simple HTML rating form. Not automatable — human raters
must run this; automate only the packaging.
"""
import json
import random


def build_mos_batch(system_to_wavs: dict, n_utts=20, seed=0, out_json="mos_batch.json"):
    rng = random.Random(seed)
    batch = {sys: rng.sample(wavs, min(n_utts, len(wavs))) for sys, wavs in system_to_wavs.items()}
    with open(out_json, "w") as f:
        json.dump(batch, f, indent=2)
    return batch
