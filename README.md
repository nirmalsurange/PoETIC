# PoETIC: A Re-framing of Context Dependent Emotion Detection

Official repository for the paper:

> **PoETIC: A Re-framing of Context Dependent Emotion Detection**  
> **Nirmal Surange, Manish Shrivastava**  
> Language Technologies Research Center (LTRC), IIIT Hyderabad

---

## Overview

Emotion classification datasets traditionally assume that the emotion of an utterance is completely encoded within the text itself. However, many utterances are inherently **context-dependent**, and their emotional interpretation changes depending on the surrounding situation.

PoETIC introduces a new perspective on emotion understanding by reformulating emotion classification as a **context-dependency classification task**.

Instead of asking:

> *"What emotion does this sentence express?"*

PoETIC asks:

> *"Can this sentence express a given emotion without context, only with additional context, or not at all?"*

To support this task, we present **PoETIC (Plausibility of Emotion Tags for Imagined Contexts)**, a human-annotated benchmark built upon the GoEmotions dataset.

---

## Motivation

Current text-only emotion datasets ignore an important property of language:

> **Emotion is often determined by context rather than the sentence alone.**

For example,

> *"I can't believe this happened."*

Depending on the surrounding context, this sentence may plausibly express:

- Joy
- Surprise
- Sadness
- Anger

Traditional datasets assign only a single emotion label, whereas PoETIC explicitly models this contextual ambiguity.

---

## Context-Dependency Labels

Each **(Utterance, Emotion)** pair is annotated using one of three labels.

| Label | Meaning |
|--------|---------|
| **Emotion-Obvious (EO)** | The emotion is clearly expressed without additional context. |
| **Emotion-Plausible (EP)** | The emotion can be expressed if an appropriate context is imagined. |
| **Emotion-Implausible (EI)** | The utterance cannot plausibly express the emotion under any reasonable context. |

---

## Repository Structure

```text
PoETIC/
│
├── Dataset/
│   └── poetic.csv
│
├── Scripts/
│   ├── evaluation/
│   └── inference/
│
├── Results/
│   ├── tables/
│   └── figures/
│
└── README.md
```

---

## Dataset

The dataset consists of human annotations over utterance-emotion pairs.

Each record contains:

- Text-id
- ID
- Utterance
- Target emotion
- Context-dependency label

The dataset is available as

```text
Dataset/poetic.csv
```

---

## Benchmark Statistics

| Property | Value |
|----------|------:|
| Source Dataset | GoEmotions Test Set |
| Selected Sentences | 1,000 |
| Emotion Categories | 7 |
| Annotated Pairs | 7,000 |
| Annotation Labels | EO / EP / EI |
| Inter-Annotator Agreement | Krippendorff's α = 0.62 |

---

## Repository Contents

This repository contains:

- PoETIC benchmark dataset
- Inference scripts with Prompt templates
- Evaluation scripts
- Figures and tables from the paper

---

## Experimental Results

We evaluate several proprietary and open-source language models on the proposed benchmark, including:

- GPT-5-mini
- Gemini-2.5 Flash
- Llama-3-70B
- Qwen3-8B
- Llama-3-8B
- RoBERTa baseline

Detailed evaluation tables and figures are available in the `Results/` directory.

---
## Code

The repository includes scripts for:

- **Inference** — running the evaluated models on the PoETIC benchmark. See [`scripts/inference/`](scripts/inference/).
- **Evaluation** — computing the reported classification metrics and generating evaluation results. See [`scripts/evaluation/`](scripts/evaluation/).

---

## Citation

If you use PoETIC in your work, please cite:

```bibtex
@inproceedings{surange2026poetic,
  author    = {Surange, Nirmal and Shrivastava, Manish},
  title     = {{PoETIC}: A Re-framing of Context Dependent Emotion Detection},
  booktitle = {Proceedings of Computational Affective Science ({CAS}) @ {LREC} 2026},
  year      = {2026},
  pages     = {201--211},
  publisher = {ELRA Language Resources Association}
}
```

---

## Paper

- 📄 [Find the paper-PDF here](PoETIC.pdf)
- 📄 ACL Anthology: *Coming Soon*

---

## License

This dataset is released under the **CC BY-NC-SA 4.0** license.

---

## Contact

**Nirmal Surange**

Language Technologies Research Center (LTRC)

IIIT Hyderabad

For questions, suggestions, or collaborations, please open an issue in this repository or contact the author directly.

---

## Acknowledgements

The PoETIC benchmark is built upon the GoEmotions dataset introduced by Demszky et al. (2020). We thank the human annotators whose careful judgments made this resource possible.
