import torch
from torch.utils.data import Dataset
from .text_processing import text_to_phoneme_ids
from .audio_processing import wav_to_mel
import torchaudio


class TTSDataset(Dataset):
    """Expects a filelist: path/to/wav.wav|transcript[|speaker_id]"""

    def __init__(self, filelist_path: str, multi_speaker: bool = False):
        self.items = []
        with open(filelist_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|")
                self.items.append(parts)
        self.multi_speaker = multi_speaker

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        parts = self.items[idx]
        wav_path, text = parts[0], parts[1]
        spk_id = int(parts[2]) if self.multi_speaker and len(parts) > 2 else 0

        wav, sr = torchaudio.load(wav_path)
        mel = wav_to_mel(wav)                       # (80, T_mel)
        phonemes = torch.LongTensor(text_to_phoneme_ids(text))

        return {
            "phonemes": phonemes,
            "mel": mel,
            "spk_id": spk_id,
        }


def collate_fn(batch):
    phon_lens = torch.LongTensor([b["phonemes"].shape[0] for b in batch])
    mel_lens = torch.LongTensor([b["mel"].shape[1] for b in batch])

    max_phon = phon_lens.max().item()
    max_mel = mel_lens.max().item()
    n_mels = batch[0]["mel"].shape[0]

    phon_pad = torch.zeros(len(batch), max_phon, dtype=torch.long)
    mel_pad = torch.zeros(len(batch), n_mels, max_mel)
    spk_ids = torch.zeros(len(batch), dtype=torch.long)

    for i, b in enumerate(batch):
        phon_pad[i, : b["phonemes"].shape[0]] = b["phonemes"]
        mel_pad[i, :, : b["mel"].shape[1]] = b["mel"]
        spk_ids[i] = b["spk_id"]

    return {
        "phonemes": phon_pad,
        "phon_lens": phon_lens,
        "mel": mel_pad,
        "mel_lens": mel_lens,
        "spk_ids": spk_ids,
    }
