# 🚨 Disaster Response Tweet Classification

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-4.48%2B-yellow.svg)](https://huggingface.co/transformers/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **Academic Project Report & Implementation**  
> **Course:** CSE 4122 — Natural Language Processing Laboratory  
> **Institution:** Department of Computer Science and Engineering, Khulna University of Engineering & Technology (KUET), Bangladesh  
> **Authors:** Md. Nayeem (Roll: 2107050) & MD Jahid Hasan Jim (Roll: 2107054)  
> **Supervised by:** Dr. K. M. Azharul Hasan (Professor) & Md Nazirulhasan Shawon (Assistant Professor)

---

## 📌 Executive Summary

During major natural and anthropogenic emergencies (hurricanes, earthquakes, flash floods, wildfires), social media platforms like Twitter/X serve as critical real-time communication backbones. However, crisis response agencies face an acute information triage bottleneck: high-velocity disaster streams are overwhelmed by noise, casual chatter, and emotional commentary.

This project delivers an **end-to-end, multi-paradigm NLP benchmark and decision-support system** for 10-class humanitarian disaster tweet classification on the benchmark **HumAID** dataset (**76,484 annotated tweets** across 19 disaster events). 

### Key Achievements:
* **Domain-Specific Noise Sanitization:** An 8-stage cleaning pipeline featuring crisis emoji-to-text semantic translation, CamelCase hashtag segmentation, and contraction expansion, reducing vocabulary sparsity by **40.8%**.
* **Comprehensive 16-Model Multi-Paradigm Benchmark:** Designed, trained, and evaluated **6 distinct model families** spanning Classical ML, Dense Word Embeddings (Word2Vec, FastText, GloVe) paired with Recurrent Networks (BiRNN, BiGRU, BiLSTM, Stacked BiLSTM), and Pretrained Contextual Transformers (BERT Base and RoBERTa Base).
* **State-of-the-Art Performance:** **RoBERTa-Base** established the champion benchmark with **0.7619 Test Macro F1** and **0.7831 Test Accuracy**, with exceptional resilience on rare classes (e.g., **79.71% F1** on *Missing or Found People*, despite comprising only 0.47% of the dataset).
* **Sequence Length Truncation Ablation:** Empirical sweep over $L \in \{32, 48, 64, 128\}$, identifying $L = 48$ as the optimal trade-off (**99.4% token coverage**, highest Macro F1, and **1.40× training speedup** over $L=128$).
* **Weighted Soft-Voting Ensemble:** Combines model predictions weighted by validated test Macro F1 scores ($P_{\text{ens}}(c) = \sum w_m P_m(c) / \sum w_m$).
* **Interactive Crisis Triage Web Application:** Full-featured Streamlit dashboard (`app.py`) for real-time single-tweet classification, multi-model consensus breakdown, test split batch evaluation, and benchmark exploration.

---

## 📊 Dataset Overview: HumAID Benchmark

The **HumAID** (*Human-Annotated Disaster Incidents Data from Twitter*) corpus is curated by the Qatar Computing Research Institute (QCRI) and covers 19 disaster events between 2016 and 2019.

### Dataset Partitioning (Stratified)
* **Training Set (70%):** 53,531 tweets
* **Validation Set (10%):** 7,793 tweets
* **Test Set (20%):** 15,160 tweets
* **Total:** 76,484 tweets

### Target Humanitarian Classes & Imbalance Profile
HumAID exhibits extreme class imbalance (a **58.5:1 ratio** between the largest and rarest class):

| # | Humanitarian Category | Operational Scope / Description | Train | Val | Test | Total |
|---|---|---|---|---|---|---|
| 1 | **Caution & Advice** | Early warnings, safety guidance, weather alerts | 3,763 | 550 | 1,070 | 5,383 |
| 2 | **Displaced People & Evacuations** | Evacuation orders, temporary shelters, refugees | 2,786 | 406 | 790 | 3,982 |
| 3 | **Infrastructure & Utility Damage** | Collapsed bridges, damaged roads, power blackouts | 5,710 | 831 | 1,617 | 8,158 |
| 4 | **Injured or Dead People** | Casualties, injury reports, confirmed fatalities | 5,108 | 744 | 1,447 | 7,299 |
| 5 | **Missing or Found People** | Missing person alerts, reunions, found notices | 256 | 36 | 72 | 364 |
| 6 | **Not Humanitarian** | Casual chats, political commentary, irrelevant spam | 4,395 | 640 | 1,245 | 6,280 |
| 7 | **Other Relevant Information** | Broad situation updates, news broadcasts | 8,499 | 1,237 | 2,407 | 12,143 |
| 8 | **Requests & Urgent Needs** | Urgent calls for food, water, medical aid, rescue | 1,838 | 267 | 521 | 2,626 |
| 9 | **Rescue, Volunteering & Donations** | Donation drives, NGO relief operations, volunteer work | 14,891 | 2,194 | 4,219 | 21,304 |
| 10 | **Sympathy & Support** | Condolences, prayers, emotional solidarity messages | 6,285 | 888 | 1,772 | 8,945 |
| | **Total** | | **53,531** | **7,793** | **15,160** | **76,484** |

---

## 🧹 Preprocessing & Domain-Specific Noise Profiling

Social media crisis text contains severe syntactic noise. Our 8-stage cleaning pipeline resolved all identified noise dimensions:

```text
Raw Crisis Tweet
   │
   ├── 1. HTML Entity Decoding (html.unescape)
   ├── 2. Domain-Specific Crisis Emoji Translation (🚨 -> "emergency warning alert", 🙏 -> "prayer support")
   ├── 3. URL & User Mention Stripping (http://..., @username)
   ├── 4. CamelCase Hashtag Segmentation (#HurricaneFlorenceRelief -> "hurricane florence relief")
   ├── 5. Retweet Header Removal (RT @handle:)
   ├── 6. Contraction Expansion & Crisis Slang Normalization (pls -> "please", evac -> "evacuation")
   ├── 7. Character Elongation Normalization (pleeeaaase -> "please")
   └── 8. Whitespace & NFKD Unicode Normalization
   │
   ▼
Cleaned, Information-Dense Crisis Text (-40.8% Vocabulary Sparsity)
```

### Quantitative Profiling Before vs. After Cleaning (53,531 Training Tweets)

| Artifact / Noise Category | Raw (Before) | Cleaned (After) | Transformation Action Taken |
|---|---|---|---|
| URLs & Web Hyperlinks | 6,104 | 0 | 100.0% Stripped via Regex |
| User Mentions (`@handle`) | 33,250 | 0 | 100.0% Removed |
| Raw Hashtags (`#tag`) | 66,482 | 0 | CamelCase Segmented into Lexical Words |
| HTML Entities (`&amp;`, etc.) | 6,275 | 0 | 100.0% Unescaped to Canonical ASCII |
| Smart Quotes & Irregular Dashes | 7,215 | 0 | Standardized to ASCII Punctuation |
| Raw Emoji Pictograms | 2,867 | 0 | Translated to Crisis Semantic Keywords |
| Retweet Headers (`RT @...`) | 8,214 | 0 | 100.0% Stripped |
| Non-ASCII / Mojibake Characters | 23,900 | 0 | NFKD Unicode Normalized |
| **Unique Token Vocabulary** | **121,237** | **71,822** | **-40.8% Sparsity Reduction** |

---

## 🏆 Master Benchmark Leaderboard

All 16 model architectures evaluated on the held-out test set of **15,160 tweets**, ranked by **Test Macro F1**:

| Rank | Model Architecture | Family | Macro F1 | Accuracy | Weighted F1 | Precision | Recall |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|
| 🥇 **1** | **RoBERTa-Base (Champion)** | **Transformer** | **0.7619** | **0.7831** | **0.7783** | **0.7649** | **0.7612** |
| 🥈 **2** | **BERT-Base-Uncased** | **Transformer** | **0.7586** | **0.7792** | **0.7753** | **0.7628** | **0.7567** |
| 🥉 **3** | **FastText + BiLSTM** | **FastText Subword** | **0.7404** | **0.7627** | **0.7612** | **0.7500** | **0.7344** |
| 4 | Word2Vec + 2-Stacked BiLSTM | Word2Vec | 0.7369 | 0.7549 | 0.7513 | 0.7404 | 0.7364 |
| 5 | Word2Vec + BiLSTM | Word2Vec | 0.7329 | 0.7567 | 0.7558 | 0.7333 | 0.7375 |
| 6 | Word2Vec + BiGRU | Word2Vec | 0.7279 | 0.7516 | 0.7494 | 0.7500 | 0.7128 |
| 7 | FastText + BiRNN | FastText Subword | 0.7258 | 0.7421 | 0.7428 | 0.7361 | 0.7190 |
| 8 | Word TF-IDF + Logistic Regression | Classical ML | 0.7241 | 0.7433 | 0.7443 | 0.7112 | 0.7398 |
| 9 | Word TF-IDF + LinearSVC | Classical ML | 0.7233 | 0.7485 | 0.7450 | 0.7136 | 0.7358 |
| 10 | GloVe + BiLSTM | GloVe Pretrained | 0.7219 | 0.7538 | 0.7461 | 0.7505 | 0.7019 |
| 11 | BoW + LinearSVC | Classical ML | 0.7180 | 0.7404 | 0.7387 | 0.7133 | 0.7238 |
| 12 | BoW + Logistic Regression | Classical ML | 0.7161 | 0.7355 | 0.7378 | 0.7058 | 0.7289 |
| 13 | GloVe + BiRNN | GloVe Pretrained | 0.7100 | 0.7410 | 0.7414 | 0.7364 | 0.6926 |
| 14 | Word2Vec + BiRNN | Word2Vec | 0.7090 | 0.7406 | 0.7393 | 0.7360 | 0.6911 |
| 15 | Word TF-IDF + MultinomialNB | Classical ML | 0.6401 | 0.6883 | 0.6828 | 0.6636 | 0.6251 |
| 16 | BoW + MultinomialNB | Classical ML | 0.6309 | 0.6793 | 0.6727 | 0.6245 | 0.6483 |

---

## 🔬 Champion Model Breakdown: RoBERTa Base

### Per-Class Detailed Performance (Test Set: 15,160 Tweets)

| Humanitarian Target Category | Precision | Recall | F1-Score | Support |
|---|:---:|:---:|:---:|:---:|
| Caution & Advice | 0.6886 | 0.7150 | 0.7015 | 1,070 |
| Displaced People & Evacuations | 0.8736 | 0.9101 | 0.8915 | 790 |
| Infrastructure & Utility Damage | 0.7945 | 0.8534 | 0.8229 | 1,617 |
| Injured or Dead People | 0.9028 | 0.9440 | 0.9230 | 1,447 |
| Missing or Found People *(Rarest: 0.47%)* | 0.8333 | 0.7639 | **0.7971** | 72 |
| Not Humanitarian | 0.6371 | 0.5711 | 0.6023 | 1,245 |
| Other Relevant Information | 0.6217 | 0.5231 | 0.5681 | 2,407 |
| Requests & Urgent Needs | 0.5969 | 0.5969 | 0.5969 | 521 |
| Rescue, Volunteering & Donations | 0.8480 | 0.9123 | 0.8790 | 4,219 |
| Sympathy & Support | 0.8525 | 0.8222 | 0.8371 | 1,772 |
| **Macro Average** | **0.7649** | **0.7612** | **0.7619** | **15,160** |
| **Weighted Average** | **0.7762** | **0.7831** | **0.7783** | **15,160** |
| **Overall Accuracy** | \multicolumn{4}{c}{**78.31%**} |

---

## 🧪 Ablation Studies

### 1. Sequence Length Truncation Sweep
Ablation across sequence truncation length $L \in \{32, 48, 64, 128\}$ on transformer fine-tuning:

| Max Length ($L$) | Token Coverage | Val Macro F1 | Val Accuracy | Epoch Time | Speedup vs. $L=128$ |
|:---:|:---:|:---:|:---:|:---:|:---:|
| $L = 32$ | 96.1% | 0.7482 | 0.7712 | 312 s | $1.82\times$ |
| **$L = 48$ (Optimal)** | **99.4%** | **0.7607** | **0.7788** | **406 s** | **$1.40\times$** |
| $L = 64$ | 99.8% | 0.7563 | 0.7784 | 487 s | $1.17\times$ |
| $L = 128$ | 100.0% | 0.7548 | 0.7770 | 568 s | $1.00\times$ |

> **Takeaway:** Setting $L=48$ captures **99.4% of all tweet tokens** while achieving superior Macro F1 and a **40% throughput speedup** over the standard $L=128$.

### 2. Transformer Learning Rate Sweep
* $\eta = 3\times 10^{-5}$: **Optimal convergence** (Val Macro F1 = 0.7607, Val Acc = 0.7788).
* $\eta = 5\times 10^{-5}$: Fast convergence but minor oscillation in later epochs (Val Macro F1 = 0.7604).
* $\eta = 1\times 10^{-5}$: Underfitted within 4 epochs (Val Macro F1 = 0.7517).

---

## 💻 Interactive Crisis Triage Web Application (`app.py`)

A production-grade decision support dashboard built with **Streamlit**:

```bash
streamlit run app.py
```

### Key Modules:
1. **Live Single-Tweet Classifier:** Real-time multi-class prediction, interactive probability distributions, and preprocessed token inspection.
2. **Multi-Model Consensus Breakdown:** Side-by-side comparative inference across Classical ML, Recurrent, and Transformer models.
3. **Soft-Voting Weighted Ensemble:** Dynamically aggregates model confidence vectors weighted by benchmark Macro F1.
4. **Batch Dataset Evaluator:** Evaluates samples from the 15,160-tweet test split or custom uploaded CSVs with exportable reports.
5. **Interactive Benchmark Explorer:** In-app inspection of confusion matrices, learning curves, and per-class metrics.

---

## 📁 Repository Structure

```text
nlp-project/
├── app.py                                # Production Streamlit web application
├── dataset/                              # HumAID CSV & Parquet splits (Train, Val, Test)
│   ├── train_clean.parquet               # Cleaned Training set (53,531 rows)
│   ├── validation_clean.parquet          # Cleaned Validation set (7,793 rows)
│   └── test_clean.parquet                # Cleaned Test set (15,160 rows)
├── notebook/                             # Modular, reproducible research notebooks
│   ├── 00_eda_and_preprocessing.ipynb    # EDA & 8-stage text sanitization pipeline
│   ├── 01_classical_ml_tfidf_bow.ipynb   # BoW / TF-IDF + MNB, LinearSVC, Logistic Regression
│   ├── 02_word2vec_recurrent_models.ipynb# Word2Vec + BiRNN, BiGRU, BiLSTM, Stacked BiLSTM
│   ├── 03_fasttext_recurrent_models.ipynb# FastText subword + BiRNN, BiLSTM
│   ├── 04_glove_recurrent_models.ipynb   # GloVe Twitter + BiRNN, BiLSTM
│   ├── 05_transformer_bert.ipynb         # BERT-Base fine-tuning, sweep & evaluation
│   └── 06_transformer_roberta.ipynb      # RoBERTa-Base fine-tuning, sweep & evaluation
├── report/                               # Academic LaTeX report & project figures
│   ├── main.tex                          # Full academic report source (KUET format)
│   ├── main.pdf                          # Compiled project report PDF
│   └── figures/                          # Publication-quality plots & confusion matrices
├── results/                              # Benchmark metrics, reports, and visualization artifacts
│   ├── 01_classical_ml_tfidf_bow/        # Metrics & confusion matrices for Classical ML
│   ├── 02_word2vec_recurrent_models/     # Metrics & confusion matrices for Word2Vec
│   ├── 03_fasttext_recurrent_models/     # Metrics & confusion matrices for FastText
│   ├── 04_glove_recurrent_models/        # Metrics & confusion matrices for GloVe
│   ├── 05_transformer_bert/              # Metrics & confusion matrices for BERT
│   ├── 06_transformer_roberta/           # Metrics & confusion matrices for RoBERTa
│   ├── master_models_comparison.png      # Master 16-model benchmark comparison bar chart
│   └── v2_master_metrics_summary.csv     # Master summary table across all models
├── .gitignore                            # Excluded temporary artifacts & large weights
└── README.md                             # Project documentation
```

---

## 🚀 Quick Start Guide

### 1. Clone Repository & Setup Environment
```bash
git clone https://github.com/codernayeem/disaster-tweet-classification.git
cd disaster-tweet-classification

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Or install core packages directly: `pip install torch torchvision transformers datasets scikit-learn pandas numpy matplotlib seaborn streamlit joblib fasttext-wheel`)*

### 3. Launch the Interactive Web App
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 4. Run Notebooks
All experimental pipelines are organized sequentially in the `notebook/` directory and can be executed in Jupyter Lab, VS Code, or Kaggle GPU environments.

---

## 📚 References & Citations

1. **HumAID Dataset:** F. Alam, H. Sajjad, M. Imran, and F. Ofli, *"HumAID: Human-Annotated Disaster Incidents Data from Twitter with Deep Learning Benchmarks,"* in *Proceedings of the International AAAI Conference on Web and Social Media (ICWSM)*, vol. 15, pp. 909–920, 2021.
2. **RoBERTa:** Y. Liu, M. Ott, N. Goyal, J. Du, M. Joshi, D. Chen, O. Levy, M. Lewis, L. Zettlemoyer, and V. Stoyanov, *"RoBERTa: A Robustly Optimized BERT Pretraining Approach,"* *arXiv preprint arXiv:1907.11692*, 2019.
3. **BERT:** J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, *"BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding,"* in *Proceedings of NAACL-HLT*, pp. 4171–4186, 2019.
4. **FastText:** P. Bojanowski, E. Grave, A. Joulin, and T. Mikolov, *"Enriching Word Vectors with Subword Information,"* *Transactions of the Association for Computational Linguistics (TACL)*, vol. 5, pp. 135–146, 2017.
5. **GloVe:** J. Pennington, R. Socher, and C. D. Manning, *"GloVe: Global Vectors for Word Representation,"* in *Proceedings of EMNLP*, pp. 1532–1543, 2014.

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
