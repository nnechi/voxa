# Lightweight CNN-BiLSTM Audio-Visual ASR Pipeline

A lightweight automatic speech recognition (ASR) experiment pipeline built with PyTorch for comparing:

- audio-only transcription
- audio-visual transcription
- multiple decoding strategies:
  - greedy decoding
  - beam search
  - KenLM without a lexicon
  - KenLM with a lexicon

The project uses the **LRS2** dataset and evaluates performance with:

- **CTC Loss**
- **WER** (Word Error Rate)
- **CER** (Character Error Rate)

---

## Overview

This project explores whether a lightweight **CNN + BiLSTM** architecture can provide a reproducible ASR baseline for speech transcription and subtitling.

The main experiment compares:

- **Audio-only CNN-BiLSTM**
- **Audio-visual CNN-BiLSTM**
- **Decoder variants**
- **CNN-only audio baseline**
- optional comparison against **Whisper**

---

## Features

- character-level speech transcription
- mel spectrogram audio frontend
- optional video-frame fusion
- CTC-based training
- decoder comparisons using:
  - greedy decoding
  - CTC beam search
  - KenLM decoding
  - KenLM + lexicon decoding
- transcript output logs for qualitative analysis

---

## Project Structure

```text
.
├── main.py
├── train.py
├── dataset.py
├── sample.py
├── AudioModel.py
├── whisper_eval.py                  # optional Whisper evaluation
├── lexicon.txt                      # optional
├── kenlm.bin                        # optional
├── best_audio_model.pt              # generated after training
├── best_av_model.pt                 # generated after training
├── audio_model_greedy.txt
├── audio_model_beam.txt
├── audio_model_kenlm_nolexicon.txt
├── audio_model_kenlm_lexicon.txt
├── av_model_greedy.txt
├── av_model_beam.txt
├── av_model_kenlm_nolexicon.txt
└── av_model_kenlm_lexicon.txt
