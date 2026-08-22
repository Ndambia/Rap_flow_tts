import torch
import torch.nn as nn
from .text_encoder import TextEncoder
from .aligner import DurationPredictor, monotonic_alignment_search, expand_by_duration
from .flow_decoder import FlowMatchingDecoder


class RapFlowTTS(nn.Module):
    def __init__(self, n_vocab, n_mels=80, enc_dim=192, n_spk=0):
        super().__init__()
        self.text_encoder = TextEncoder(n_vocab, dim=enc_dim, n_mels=n_mels)
        self.duration_predictor = DurationPredictor(enc_dim)
        self.decoder = FlowMatchingDecoder(n_mels=n_mels, n_spk=n_spk)
        # theta^-: stop-gradient teacher, an EMA (or plain copy) of the decoder,
        # per Eq.(3)-(4). See training/trainer_utils.py.

    def encode(self, phonemes, phon_lens):
        h, mu_phoneme, mask = self.text_encoder(phonemes, phon_lens)
        log_dur_pred = self.duration_predictor(h.transpose(1, 2))
        return h, mu_phoneme, log_dur_pred, mask

    def build_prior(self, mu_phoneme, phon_lens, mel_lens, mel=None, training=True):
        """Returns expanded prior mu: (B, n_mels, T_mel)."""
        if training:
            # log_prior computed elsewhere for MAS (alignment_losses.py); here we assume
            # `durations` (from MAS) are passed in by the training loop.
            raise NotImplementedError("durations must be supplied by the training loop")
        else:
            durations = None  # predicted at inference, see inference/sampler.py
            raise NotImplementedError
