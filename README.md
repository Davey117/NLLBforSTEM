# Indigenous STEM Knowledge Synthesis System
### Fine-Tuned Neural Machine Translation (NMT) & Neuro-Symbolic Architecture for African STEM Education

[![Streamlit App](https://static.streamlit.io/badge_cdn/svg/white.svg)](https://nllbforstem.streamlit.app/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch_&_HuggingFace-orange.svg)](https://pytorch.org/)
[![Academic Tier](https://img.shields.io/badge/Academic--Tier-B.Sc._Final_Year_Project-blue.svg)]()

---

## 🔬 Project Overview
This project presents an advanced bidirectional machine translation and instructional synthesis platform explicitly optimized for translating complex Science, Technology, Engineering, and Mathematics (STEM) literature from English into **Yoruba (`yor_Latn`)**. 

Standard commercial translation models suffer from high **fragmentation profiles** when encountering technical scientific terminology, often resorting to awkward transliterations or dropping words entirely. This system counters that limitation by fine-tuning Meta's **NLLB-200 (No Language Left Behind)** architecture on a curated parallel corpus aligned with West African educational standards (WAEC/NECO).

Additionally, the system utilizes a **hybrid neuro-symbolic layer** to handle mathematical calculations deterministically, preventing the translation model from hallucinating numbers or operators during educational tutorials.

---

## 🛠️ Key Architectural Features
* **Custom Vocabulary Enrichment:** Injected specific high-order domain loanwords (e.g., *Ìṣiṣẹ́-oòrùn-sínú-ewé* for Photosynthesis, *kábọ̀nì dáyọ́ọ́ìdì* for Carbon Dioxide) directly into the tokenizer embedding matrix, forcing single-token alignment and stopping phrase fragmentation.
* **Parameter-Efficient Fine-Tuning (PEFT):** Implemented Low-Rank Adaptation (LoRA) with a rank ($r=32$) and scaling factor ($\alpha=64$) across all linear projection layers to maximize domain-specific downstream task alignment.
* **Deterministic Mathematical Sub-Routines:** Built robust regular expression parsing pipelines to intercept mathematical word problems, routing operations directly through a symbolic calculator thread rather than relying blindly on neural text generation.
* **Premium Executive UI:** Designed a high-contrast, professional workspace utilizing customized CSS overrides to eliminate generic template styles, creating an interface optimized for clean academic review.

---

## 📂 Repository Architecture
```text
├── app.py               # Enterprise Streamlit application code base
├── dataset.csv          # Curated parallel text corpus (English-to-Yoruba)
├── instruction.jsonl    # Instructional dataset guidelines for STEM tutor Q&A
├── requirements.txt     # Python production runtime dependencies
└── README.md            # System operational documentation manual