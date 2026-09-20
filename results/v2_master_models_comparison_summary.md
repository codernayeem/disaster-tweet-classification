# HumAID Disaster Tweet Classification: Master Benchmark Leaderboard (All 6 Suites)

**Dataset:** HumAID 10-Class Crisis Tweet Corpus (76,484 Total Tweets | 15,160 Test Tweets)

---

## 🏆 Final Benchmark Leaderboard

| Rank | Model Architecture | Family | Test Macro F1 | Test Accuracy | Test Weighted F1 | Test Precision | Test Recall | Parameter Count | Optimal Config |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **🥇 1** | **RoBERTa Base** | **Transformer (RoBERTa)** | **76.19%** | **78.31%** | **77.83%** | **76.49%** | **76.12%** | **125M** | `LR=3e-5, L=48` |
| **🥈 2** | **BERT Base Uncased** | **Transformer (BERT)** | **75.86%** | **77.92%** | **77.53%** | **76.28%** | **75.67%** | **110M** | `LR=3e-5, L=48` |
| **🥉 3** | **FastText + BiLSTM** | **FastText Subword** | **74.04%** | **76.27%** | **76.12%** | **75.00%** | **73.44%** | **4.1M** | `L=48, H=256` |
| 4 | **Word2Vec + 2-Stacked BiLSTM** | **Word2Vec** | 73.69% | 75.49% | 75.13% | 74.04% | 73.64% | 12.8M | `L=48, H=256, drop=0.3` |
| 5 | **Word2Vec + BiLSTM** | **Word2Vec** | 73.29% | 75.67% | 75.58% | 73.33% | 73.75% | 11.3M | `L=48, H=256` |
| 6 | **Word2Vec + BiGRU** | **Word2Vec** | 72.79% | 75.16% | 74.94% | 75.00% | 71.28% | 10.4M | `L=48, H=128` |
| 7 | **FastText + BiRNN** | **FastText Subword** | 72.58% | 74.21% | 74.28% | 73.61% | 71.90% | 3.6M | `L=64, H=256` |
| 8 | **Word TF-IDF + Logistic Regression** | **Classical ML** | 72.41% | 74.33% | 74.43% | 71.12% | 73.98% | N-gram | `min_df=2, C=5.0` |
| 9 | **Word TF-IDF + LinearSVC** | **Classical ML** | 72.33% | 74.85% | 74.50% | 71.36% | 73.58% | N-gram | `min_df=2, C=0.5` |
| 10 | **GloVe + BiLSTM** | **GloVe Pretrained** | 72.19% | 75.38% | 74.61% | 75.05% | 70.19% | 21.8M | `L=64, H=256` |
| 11 | **BoW + LinearSVC** | **Classical ML** | 71.80% | 74.04% | 73.87% | 71.33% | 72.38% | N-gram | `min_df=2, C=0.05` |
| 12 | **BoW + Logistic Regression** | **Classical ML** | 71.61% | 73.55% | 73.78% | 70.58% | 72.89% | N-gram | `min_df=2, C=0.5` |
| 13 | **GloVe + BiRNN** | **GloVe Pretrained** | 71.00% | 74.10% | 74.14% | 73.64% | 69.26% | 20.9M | `L=64, H=64` |
| 14 | **Word2Vec + BiRNN** | **Word2Vec** | 70.90% | 74.06% | 73.93% | 73.60% | 69.11% | 10.2M | `L=32, H=64` |
| 15 | **Word TF-IDF + MultinomialNB** | **Classical ML** | 64.01% | 68.83% | 68.28% | 66.36% | 62.51% | N-gram | `30k_trigram, alpha=0.1` |
| 16 | **BoW + MultinomialNB** | **Classical ML** | 63.09% | 67.93% | 67.27% | 62.45% | 64.83% | N-gram | `20k_bigram, alpha=1.0` |

---

## 🔍 Key Architectural Insights:
1. **RoBERTa Base Achieves Benchmark State-of-the-Art:** Dynamic byte-pair encoding (BPE) and robust pretraining without next-sentence prediction enable RoBERTa to achieve the highest Macro F1 (**76.19%**) and Test Accuracy (**78.31%**), demonstrating superior semantic comprehension on short crisis tweets.
2. **FastText Outperforms GloVe & Word2Vec in Recurrent Architectures:** Subword n-gram character n-grams allow FastText + BiLSTM (**74.04%**) to gracefully handle rare crisis misspellings, hashtags, and out-of-vocabulary entities.
3. **Stacked BiLSTM Captures Hierarchical Temporal Context:** Adding a second recurrent layer to Word2Vec BiLSTM boosted Macro F1 from 73.29% to 73.69%.
