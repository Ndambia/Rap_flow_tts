# RapFlow-TTS — Notebook Quick-Start Guide

## 📥 Step 1: Download LJSpeech Dataset

**Download URL:**
```
https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
```

| Detail | Value |
|--------|-------|
| **Size** | ~2.6 GB compressed |
| **Duration** | ~24 hours of speech |
| **Samples** | 13,100 WAV files |
| **Speaker** | Single female (LJ) |
| **Sample Rate** | 22,050 Hz |
| **License** | Public Domain ✅ |

**Extract to:**
```
Rap_flow_tts/
└── data/
    └── LJSpeech-1.1/        ← extract here
        ├── metadata.csv
        └── wavs/
            ├── LJ001-0001.wav
            └── ...
```

---

## ⚙️ Step 2: Install Dependencies

```bash
# Create conda/venv environment (recommended)
conda create -n rapflow python=3.10 -y
conda activate rapflow

# Install PyTorch (with GPU support)
pip install torch>=2.1 torchaudio>=2.1 --index-url https://download.pytorch.org/whl/cu118

# Install remaining requirements
pip install -r requirements.txt

# Install Jupyter
pip install jupyter ipywidgets
```

### espeak-ng (for phonemization)
On Windows:
1. Download the installer from: https://github.com/espeak-ng/espeak-ng/releases
2. Install and add to PATH
3. **Note:** If espeak-ng is not installed, the notebook automatically uses a character-level fallback tokenizer — you can still run everything for demo purposes.

---

## 🚀 Step 3: Preprocess LJSpeech

```bash
cd Rap_flow_tts

# Preprocess and verify the dataset
python preprocess_and_verify.py --data_dir data/LJSpeech-1.1 --out_dir data/processed/ljspeech --verify_audio
```

This writes:
- `data/processed/ljspeech/train_filelist.txt` (13,000 clips)
- `data/processed/ljspeech/val_filelist.txt` (100 clips)

---

## 📓 Step 4: Open the Notebook

```bash
jupyter notebook rapflow_tts_presentation.ipynb
```

Or in VS Code: open `rapflow_tts_presentation.ipynb` directly.

---

## 🎯 Training Configuration

Edit the epoch counts in the notebook cells to match full training:

| Stage | Epochs (LJSpeech) | Epochs (VCTK) | What it trains |
|-------|-------------------|---------------|----------------|
| Stage 1 | 700 | 500 | Text Encoder + Aligner + Flow Decoder (straight-flow) |
| Stage 2 | 700 | 500 | Flow Decoder consistency (EMA teacher) |
| Stage 3 | 150 | 50 | Adversarial refinement (LSGAN discriminator) |

---

## 🔊 HiFi-GAN Vocoder (for real audio synthesis)

Download the pretrained HiFi-GAN V1 checkpoint:
- https://github.com/jik876/hifi-gan
- Model: `generator_v1` (trained on LJSpeech)

Without HiFi-GAN, the notebook uses Griffin-Lim reconstruction for demo audio (lower quality but works instantly).

---

## 📊 Expected Results (full training on LJSpeech)

| Metric | Value |
|--------|-------|
| MOS (naturalness) | ~4.2 / 5.0 |
| WER | ~5–6% |
| RTF (V100, NFE=2) | ~0.006 |
| Inference steps | 2 (ultra-fast!) |

---

## 🗂 Project Structure

```
Rap_flow_tts/
├── rapflow_tts_presentation.ipynb    ← Main notebook (start here!)
├── preprocess_and_verify.py          ← Dataset setup helper
├── requirements.txt
├── configs/
│   ├── ljspeech.yaml                 ← Single-speaker config
│   └── vctk.yaml                     ← Multi-speaker config
├── data/
│   ├── dataset.py                    ← TTSDataset + collate_fn
│   ├── audio_processing.py           ← wav_to_mel (log mel-spec)
│   └── text_processing.py            ← G2P + char fallback
├── model/
│   ├── rapflow_tts.py                ← Main model wrapper
│   ├── text_encoder.py               ← FFT Transformer encoder
│   ├── aligner.py                    ← MAS + Duration Predictor
│   ├── flow_decoder.py               ← U-Net velocity field
│   └── discriminator.py              ← LSGAN discriminator
├── training/
│   ├── train_stage1.py               ← Straight-flow pre-training
│   ├── train_stage2.py               ← Consistency FM + EMA
│   └── train_stage3_adversarial.py   ← GAN refinement
├── losses/
│   ├── consistency_fm.py             ← L_sf + L_vc objectives
│   ├── alignment_losses.py           ← Prior + Duration losses
│   └── adversarial.py                ← LSGAN losses
├── inference/
│   ├── sampler.py                    ← Euler ODE sampler
│   └── synthesize.py                 ← End-to-end synthesis
└── schedules/
    └── delta_scheduling.py           ← δ annealing schedule
```
