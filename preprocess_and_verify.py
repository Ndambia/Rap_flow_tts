"""
Preprocess LJSpeech into filelists and verify the setup.
Run this AFTER extracting LJSpeech-1.1 to data/LJSpeech-1.1/

Usage:
    cd Rap_flow_tts
    python preprocess_and_verify.py --data_dir data/LJSpeech-1.1 --out_dir data/processed/ljspeech
"""
import sys
import argparse
from pathlib import Path

# Make sure the project root is in the path
ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description="Preprocess LJSpeech + verify the pipeline")
    parser.add_argument("--data_dir", default="data/LJSpeech-1.1",
                        help="Path to the extracted LJSpeech-1.1 folder")
    parser.add_argument("--out_dir", default="data/processed/ljspeech",
                        help="Output directory for filelists")
    parser.add_argument("--val_split", type=int, default=100,
                        help="Number of samples reserved for validation")
    parser.add_argument("--verify_audio", action="store_true",
                        help="Load and mel-transform 5 samples to verify audio pipeline")
    args = parser.parse_args()

    data_dir = ROOT / args.data_dir
    out_dir  = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = data_dir / "metadata.csv"
    if not metadata_path.exists():
        print(f"\n❌  metadata.csv not found at {metadata_path}")
        print(f"\n📥  Please download LJSpeech-1.1 from:")
        print(f"     https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2")
        print(f"\n   Then extract it so the structure looks like:")
        print(f"     Rap_flow_tts/data/LJSpeech-1.1/metadata.csv")
        print(f"     Rap_flow_tts/data/LJSpeech-1.1/wavs/LJ001-0001.wav")
        sys.exit(1)

    # ── Parse metadata ────────────────────────────────────────────────────────
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
                entries.append((str(wav_path.resolve()), normalized_text))

    print(f"✅  Found {len(entries)} valid WAV + transcript pairs")

    # ── Split ─────────────────────────────────────────────────────────────────
    val_entries   = entries[:args.val_split]
    train_entries = entries[args.val_split:]
    print(f"   Train: {len(train_entries)} | Val: {len(val_entries)}")

    # ── Write filelists ───────────────────────────────────────────────────────
    train_fl = out_dir / "train_filelist.txt"
    val_fl   = out_dir / "val_filelist.txt"

    with open(train_fl, "w", encoding="utf-8") as f:
        for wav_path, text in train_entries:
            f.write(f"{wav_path}|{text}\n")

    with open(val_fl, "w", encoding="utf-8") as f:
        for wav_path, text in val_entries:
            f.write(f"{wav_path}|{text}\n")

    print(f"✅  Filelists written to {out_dir}")

    # ── Verify audio pipeline ─────────────────────────────────────────────────
    if args.verify_audio:
        import torch
        import torchaudio
        from data.audio_processing import wav_to_mel, MEL_CONFIG
        from data.text_processing import text_to_phoneme_ids

        print(f"\n🔍  Verifying audio pipeline on 5 samples...")
        for i, (wav_path, text) in enumerate(train_entries[:5]):
            wav, sr = torchaudio.load(wav_path)
            if sr != MEL_CONFIG["sample_rate"]:
                wav = torchaudio.functional.resample(wav, sr, MEL_CONFIG["sample_rate"])
            mel = wav_to_mel(wav)
            ids = text_to_phoneme_ids(text)
            print(f"   [{i+1}] mel={mel.shape}, phonemes={len(ids)}, text='{text[:50]}...'")

        print("\n✅  Audio pipeline OK!")

    print("\n🎉  Preprocessing complete!")
    print(f"   Run the notebook: rapflow_tts_presentation.ipynb")
    print(f"   Set TRAIN_FILELIST = '{train_fl}'")
    print(f"   Set VAL_FILELIST   = '{val_fl}'")


if __name__ == "__main__":
    main()
