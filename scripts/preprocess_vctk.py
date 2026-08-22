"""Preprocess VCTK dataset into training filelists.

Usage:
    python scripts/preprocess_vctk.py --data_dir /path/to/VCTK-Corpus --out_dir data/processed/vctk

Expected VCTK structure:
    VCTK-Corpus/
    ├── txt/
    │   ├── p225/
    │   │   ├── p225_001.txt
    │   │   └── ...
    │   └── ...
    └── wav48/
        ├── p225/
        │   ├── p225_001.wav
        │   └── ...
        └── ...
"""
import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Preprocess VCTK")
    parser.add_argument("--data_dir", required=True, help="Path to VCTK-Corpus directory")
    parser.add_argument("--out_dir", default="data/processed/vctk", help="Output directory")
    parser.add_argument("--val_per_speaker", type=int, default=2,
                        help="Number of validation samples per speaker")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    txt_dir = data_dir / "txt"
    wav_dir = data_dir / "wav48"  # VCTK uses 48kHz wavs

    if not wav_dir.exists():
        wav_dir = data_dir / "wav48_silence_trimmed"  # alternate naming
    assert wav_dir.exists(), f"wav directory not found at {wav_dir}"

    # Collect speaker IDs
    speakers = sorted([d.name for d in txt_dir.iterdir() if d.is_dir()])
    speaker_to_id = {spk: i for i, spk in enumerate(speakers)}
    print(f"Found {len(speakers)} speakers")

    train_entries = []
    val_entries = []

    for spk in speakers:
        spk_txt_dir = txt_dir / spk
        spk_wav_dir = wav_dir / spk
        spk_id = speaker_to_id[spk]

        txt_files = sorted(spk_txt_dir.glob("*.txt"))
        spk_entries = []

        for txt_file in txt_files:
            text = txt_file.read_text(encoding="utf-8").strip()
            stem = txt_file.stem

            # Try different wav naming conventions
            wav_path = None
            for suffix in [".wav", "_mic1.flac", "_mic2.flac"]:
                candidate = spk_wav_dir / f"{stem}{suffix}"
                if candidate.exists():
                    wav_path = candidate
                    break

            if wav_path is not None:
                spk_entries.append((str(wav_path), text, str(spk_id)))

        # Split per speaker
        val_count = min(args.val_per_speaker, len(spk_entries))
        val_entries.extend(spk_entries[:val_count])
        train_entries.extend(spk_entries[val_count:])

    # Write filelists (format: wav_path|text|speaker_id)
    train_filelist = out_dir / "train_filelist.txt"
    val_filelist = out_dir / "val_filelist.txt"

    with open(train_filelist, "w", encoding="utf-8") as f:
        for wav_path, text, spk_id in train_entries:
            f.write(f"{wav_path}|{text}|{spk_id}\n")

    with open(val_filelist, "w", encoding="utf-8") as f:
        for wav_path, text, spk_id in val_entries:
            f.write(f"{wav_path}|{text}|{spk_id}\n")

    # Write speaker map
    speaker_map_path = out_dir / "speaker_map.txt"
    with open(speaker_map_path, "w", encoding="utf-8") as f:
        for spk, sid in speaker_to_id.items():
            f.write(f"{spk}|{sid}\n")

    print(f"Train: {len(train_entries)} | Val: {len(val_entries)}")
    print(f"Speaker map: {speaker_map_path}")
    print(f"Filelists written to {out_dir}")


if __name__ == "__main__":
    main()
