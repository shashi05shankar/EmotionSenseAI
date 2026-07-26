# Phase 1: SOTA, Datasets & Model Reference

**Project:** EmotionSense AI
**Date:** 2026-07-26

> Reference sheet for the literature and the practical building blocks (datasets +
> pretrained models). Use this to set realistic accuracy targets and to pick which
> corpora and backbones to actually build on given Colab/Kaggle (free-GPU) compute.

---

## 1. State-of-the-art accuracy (calibrates our Phase 0.5 targets)

| Dataset | Approach | Metric | Score | Notes |
|---------|----------|--------|-------|-------|
| IEMOCAP (4-cls) | Adapted wav2vec2 / HuBERT | Weighted Acc | **79.6%** | Current strong SOTA, speaker-independent |
| IEMOCAP (4-cls) | Fine-tuned wav2vec2 + neural CDE | WA / UA | 73.4% / 74.2% | PLOS One, Feb 2025 |
| IEMOCAP (4-cls) | HuBERT-large as **frozen** extractor (SUPERB) | Weighted Acc | 67.6% | Frozen = no fine-tune; our realistic floor |
| RAVDESS (8-cls) | HuBERT X-large + SVM | Acc | 82.6% | Frozen features + classical head |
| RAVDESS | **Distil-HuBERT** | Acc | **87.0%** | Beats wav2vec2, small model — ideal for our compute |
| RAVDESS (8-cls) | Multi-head attention CNN (repo) | Acc | ~92% | Speaker-dependent — inflated, note caveat |

**Reality check for Phase 0.5:** the roadmap's "≥90% accuracy" goal is only realistic
on RAVDESS with **speaker-dependent** splits (which over-report). Honest,
**speaker-independent** targets:
- RAVDESS 8-class: **80–87%** (Distil-HuBERT territory) — set target **≥83%**.
- IEMOCAP 4-class: **70–79%** — set target **≥72%**.
- Cross-corpus (train RAVDESS → test CREMA-D): expect a large drop (often 40–55%);
  *measuring and reporting this honestly is a differentiator*, not a failure.

**Key modeling insight:** **Distil-HuBERT** delivers near-HuBERT accuracy at a
fraction of the size and runs in real time — the best fit for free-tier GPUs. Prefer
it over full wav2vec2-large for our transformer track.

---

## 2. Datasets (English-first, free/accessible)

| Dataset | Size | Speakers | Emotions | Access | Role in project |
|---------|------|----------|----------|--------|-----------------|
| **RAVDESS** | 1,440 utts | 24 actors | 8 (neutral, calm, happy, sad, angry, fearful, disgust, surprise) | Free (Zenodo) | **Primary** benchmark |
| **TESS** | 2,800 clips | 2 female | 7 | Free (Kaggle/UofT) | Augment; adds female-heavy data |
| **CREMA-D** | 7,442 clips | 91 actors (diverse) | 6 | Free (GitHub) | **Cross-corpus test set** (best diversity) |
| **SAVEE** | ~480 clips | 4 male | 7 | Free (registration) | Small; male-only add |
| **EMO-DB** | ~500 clips | 10 | 7 | Free | German — for multilingual stretch only |
| IEMOCAP | ~12h | 10 | 4 (conventional) | **Licence request** | Research benchmark; gated, slower to get |
| MELD | ~13k utts | multi (Friends TV) | 7 | Free | Multimodal/real-world stretch |

**Recommended dataset plan:**
- **Core:** RAVDESS + TESS + SAVEE unified to a common label set (train/val, speaker-independent split).
- **Cross-corpus generalization test:** hold out **CREMA-D** entirely — never train on it.
- **Stretch:** IEMOCAP (once licence arrives), MELD (multimodal).

Label harmonization is required — datasets disagree on "calm" vs "neutral", "disgust"
inclusion, etc. A shared 7-class taxonomy (angry, happy, sad, neutral, fearful,
disgust, surprise) is the pragmatic common denominator.

---

## 3. Pretrained backbones (HuggingFace) — what to actually use

| Model | Arch | Downloads | Trained on | Use in project |
|-------|------|-----------|-----------|----------------|
| `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` | wav2vec2 | 596k | MSP-Podcast | Dimensional (arousal/valence) stretch |
| `speechbrain/emotion-recognition-wav2vec2-IEMOCAP` | wav2vec2 | 61k | IEMOCAP | Baseline transformer, 4-cls |
| `ehcalabres/wav2vec2-lg-xlsr-en-SER` | wav2vec2-XLSR | 15k | RAVDESS | Direct 8-cls baseline to beat/match |
| **`ntu-spml/distilhubert`** (+ our head) | Distil-HuBERT | — | general | **Recommended fine-tune backbone** |
| `emotion2vec/emotion2vec_plus_base` | emotion2vec | — | multi | Frozen-embedding + linear head (very cheap) |
| `firdhokk/…whisper-large-v3` | Whisper | 12k | multi | Too heavy for real-time; skip for serving |

**Two viable transformer strategies on free GPU:**
1. **Frozen embeddings + light head** (emotion2vec / HuBERT features → MLP or SVM).
   Cheap, fast, no fine-tuning — great baseline, matches the 67–82% frozen numbers.
2. **Fine-tune Distil-HuBERT** on unified RAVDESS+TESS. Fits in Colab/Kaggle sessions
   with checkpointing; targets the mid-80s.

---

## 4. Notable benchmarks/tools to borrow methodology from

- **SUPERB / EMO-SUPERB** — standard frozen-feature evaluation protocol (5-fold, drop
  imbalanced classes to 4). Adopt its speaker-independent discipline.
- **EmoBox** — 32 datasets × 14 languages harness; reference for cross-corpus rigor.
- **emotion2vec** (ACL 2024 Findings) — SSL emotion representation; linear-probe SOTA.

---

## Sources

- [Fine-tuned Wav2vec2.0 + neural CDE (PLOS One, 2025)](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0318297)
- [Distilled HuBERT cross-corpus study (arXiv 2024)](https://arxiv.org/pdf/2512.23435)
- [Fine-tuned wav2vec2/HuBERT benchmark (arXiv 2111.02735)](https://arxiv.org/pdf/2111.02735)
- [emotion2vec (ACL 2024 Findings)](https://aclanthology.org/2024.findings-acl.931/)
- [EmoBox: Multilingual Multi-corpus SER (Interspeech 2024)](https://www.isca-archive.org/interspeech_2024/ma24b_interspeech.pdf)
- [superb/wav2vec2-base-superb-er](https://huggingface.co/superb/wav2vec2-base-superb-er)
- [SER on MELD & RAVDESS using CNN (MDPI 2025)](https://www.mdpi.com/2078-2489/16/7/518)
- [Self-supervised SER with Distil-HuBERT (Springer 2024)](https://link.springer.com/article/10.1007/s11518-024-5607-y)
