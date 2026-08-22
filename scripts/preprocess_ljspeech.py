"""Preprocess LJSpeech dataset into training filelists.

Usage:
    python scripts/preprocess_ljspeech.py --data_dir /path/to/LJSpeech-1.1 --out_dir data/processed/ljspeech

Expected LJSpeech structure:
    LJSpeech-1.1/
    ├── metadata.csv          # id|transcription|normalized_transcription
    └── wavs/
        ├── LJ001-0001.wav
        └── ...
"""
import argparse
import os
from pathlib import Path
import torchaudio
from data.audio_processing import wav_to_mel, MEL_CONFIG
import torch


def main():
    parser = argparse.ArgumentParser(description="Preprocess LJSpeech")
    parser.add_argument("--data_dir", required=True, help="Path to LJSpeech-1.1 directory")
    parser.add_argument("--out_dir", default="data/processed/ljspeech", help="Output directory")
    parser.add_argument("--val_split", type=int, default=100, help="Number of validation samples")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = data_dir / "metadata.csv"
    assert metadata_path.exists(), f"metadata.csv not found at {metadata_path}"

    entries = []
    with open(metadata_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) >= 3:
                uid, _, normalized_text = parts[0], parts[1], parts[2]
            elif len(parts) == 2:
                uid, normalized_text = parts[0], parts[1]
            else:
                continue
            wav_path = data_dir / "wavs" / f"{uid}.wav"
            if wav_path.exists():
                entries.append((str(wav_path), normalized_text))

    print(f"Found {len(entries)} valid entries")

    # Split into train/val
    val_entries = entries[:args.val_split]
    train_entries = entries[args.val_split:]

    # Write filelists
    train_filelist = out_dir / "train_filelist.txt"
    val_filelist = out_dir / "val_filelist.txt"

    with open(train_filelist, "w", encoding="utf-8") as f:
        for wav_path, text in train_entries:
            f.write(f"{wav_path}|{text}\n")

    with open(val_filelist, "w", encoding="utf-8") as f:
        for wav_path, text in val_entries:
            f.write(f"{wav_path}|{text}\n")

    print(f"Train: {len(train_entries)} | Val: {len(val_entries)}")
    print(f"Filelists written to {out_dir}")

    # Verify a sample
    if train_entries:
        wav_path = train_entries[0][0]
        wav, sr = torchaudio.load(wav_path)
        if sr != MEL_CONFIG["sample_rate"]:
            wav = torchaudio.functional.resample(wav, sr, MEL_CONFIG["sample_rate"])
        mel = wav_to_mel(wav)
        print(f"Sample mel shape: {mel.shape} (n_mels={mel.shape[0]}, frames={mel.shape[1]})")


if __name__ == "__main__":
    main()
