# RapFlow-TTS Presentation & Training Notebook Implementation Plan

This plan details the design and implementation of a comprehensive, presentation-ready Jupyter notebook (`rapflow_tts_presentation_and_training.ipynb`) and supporting setup utilities for **RapFlow-TTS**. The notebook will serve both as an interactive training/experimentation environment and as a visual demonstration tool for your MSc presentation.

---

## Key Features & Structure

### 1. Dataset Selection & Setup Guide
- **Recommended Dataset: LJSpeech 1.1** (Single-Speaker, ~24 hours, ~2.6 GB download).
  - Download URL: `https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2`
  - Ideal for single-speaker TTS, clean audio, easy setup.
- **Alternative Dataset: VCTK** (Multi-Speaker, ~109 speakers, ~11 GB download).
  - Suitable for multi-speaker adaptation.
- **Micro/Synthetic Dataset Auto-Generator**:
  - Auto-creates a 5-sample dummy audio & transcript dataset in `data/demo_dataset` directly from the notebook.
  - Ensures the notebook can be executed **immediately** during live presentation without waiting for full dataset downloads.

### 2. Architecture & Stepwise Visualizations (For MSc Presentation)
- **Model Architecture Overview**: Mermaid diagram & theoretical explanation of Flow Matching, Monotonic Alignment Search (MAS), and Duration Prediction.
- **Step 1: Text Processing & Phoneme Mapping**:
  - Text to IPA phonemes visualization and symbol distribution histogram.
  - Windows fallback handling (handles cases where `espeak-ng` binary is not in Windows system PATH).
- **Step 2: Monotonic Alignment Search (MAS) Visualizer**:
  - Interactive heatmap plot showing the hard alignment matrix $A \in \{0, 1\}^{T_{mel} \times T_{txt}}$ between phonemes and mel-spectrogram time frames.
- **Step 3: Step-by-Step Flow Matching Generation ($t=0 \to t=1$)**:
  - Visualizing the vector field transformation at intermediate time steps (e.g., $t = 0.0, 0.25, 0.50, 0.75, 1.00$).
  - Shows how Gaussian noise progressively shapes into clean speech mel-spectrograms.
- **Step 4: Mel-Spectrogram & Audio Synthesis**:
  - Comparison of original vs generated Mel-Spectrograms.
  - Audio rendering via `IPython.display.Audio` widget for live presentation listening.

### 3. Model Training & Evaluation Cells
- **Stage 1 Training Loop**:
  - Joint optimization of Flow Matching loss, Duration Loss, and Prior Loss.
  - Live progress bars (`tqdm`) and loss curve plotting (Loss vs Epochs).
- **Stage 2 & Stage 3 Training Loops**:
  - Consistency Flow Matching and Adversarial refinement steps.
- **Inference & Custom Speech Synthesis**:
  - Interactive widget/cell to type any text prompt and synthesize audio on demand.

---

## User Review Required

> [!NOTE]
> **Dataset Recommendation**: We recommend **LJSpeech-1.1** as the primary dataset. It is ~2.6 GB in size and contains 13,100 short audio clips of a single speaker with transcriptions. 
> To test the notebook immediately before downloading the full dataset, the notebook includes an **automatic mini synthetic dataset generator** so you can run all visualizations and training cells instantly.

> [!IMPORTANT]
> **Windows Dependencies**: Phonemization uses `phonemizer` with `espeak-ng`. On Windows systems without `espeak-ng` pre-installed, we will include a fallback character tokenizer so that the presentation notebook runs smoothly without crash risks.

---

## Proposed Changes

### [Component 1] Notebook & Demonstrations
#### [NEW] [rapflow_tts_presentation_and_training.ipynb](file:///C:/Users/Stuxs/Documents/MSC%20SEM%201/MSc%20Sem%202/Rap_flow_tts/rapflow_tts_presentation_and_training.ipynb)
- Interactive Jupyter notebook containing:
  1. Title & Presentation Introduction
  2. Dataset Download & Setup Guide (LJSpeech & VCTK)
  3. Micro Synthetic Dataset Auto-Creation Cell
  4. Step 1: Text Preprocessing & Phoneme Visualizer
  5. Step 2: Monotonic Alignment Search (MAS) Alignment Matrix Heatmap
  6. Step 3: Vector Field & Flow Matching ($t=0 \to t=1$) Stepwise Diffusion Visualizer
  7. Step 4: Training Pipeline (Stage 1, Stage 2, Stage 3) with Live Loss Curves
  8. Step 5: Audio Synthesis & Playback Widget

### [Component 2] Robustness & Utility Helpers
#### [MODIFY] [text_processing.py](file:///C:/Users/Stuxs/Documents/MSC%20SEM%201/MSc%20Sem%202/Rap_flow_tts/data/text_processing.py)
- Add safety fallback when `espeak` is not installed on host OS, ensuring seamless character-level mapping so text tokenization never fails.

---

## Verification Plan

### Automated Tests
- Run synthetic data generation cell to create dummy sample wavs and transcripts.
- Execute notebook cells sequentially via python/jupyter to verify zero errors in data loading, alignment calculation, flow matching sampling, and spectrogram visualization.

### Manual Verification
- Verify plot outputs (MAS heatmap, Flow vector grid, Spectrograms) render clearly with readable titles, labels, and colorbars suitable for presentation slides.
- Test `IPython.display.Audio` widget for audio synthesis.
