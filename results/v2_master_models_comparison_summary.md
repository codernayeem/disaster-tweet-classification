# Master Model Comparison Leaderboard (HumAID Dataset - 24 Models)

Comprehensive comparative evaluation across 6 model families evaluated on the cleaned test set (15,160 tweets).

| Rank | Model | Category | Test Macro F1 | Test Accuracy | Test Weighted F1 | Test Macro Precision | Test Macro Recall |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **1** | **BERT Base Uncased** | `Transformer (BERT)` | **0.7540** | **0.7710** | 0.7678 | 0.7353 | 0.7815 |
| **2** | **RoBERTa Base** | `Transformer (RoBERTa)` | **0.7529** | **0.7710** | 0.7674 | 0.7315 | 0.7860 |
| **3** | **Word TF-IDF + LogisticRegression** | `Classical ML` | **0.7172** | **0.7363** | 0.7387 | 0.6983 | 0.7444 |
| **4** | **Char TF-IDF + LinearSVC** | `Classical ML` | **0.7120** | **0.7362** | 0.7344 | 0.7025 | 0.7239 |
| **5** | **Char TF-IDF + LogisticRegression** | `Classical ML` | **0.7112** | **0.7336** | 0.7366 | 0.6874 | 0.7500 |
| **6** | **BoW + LogisticRegression** | `Classical ML` | **0.7085** | **0.7254** | 0.7277 | 0.6991 | 0.7198 |
| **7** | **Word TF-IDF + LinearSVC** | `Classical ML` | **0.7080** | **0.7301** | 0.7279 | 0.7004 | 0.7174 |
| **8** | **Word2Vec + LSTM** | `Word2Vec` | **0.7080** | **0.7283** | 0.7257 | 0.6899 | 0.7387 |
| **9** | **FastText + 2-Stacked BiLSTM** | `FastText` | **0.7060** | **0.7245** | 0.7254 | 0.6830 | 0.7426 |
| **10** | **GloVe + BiLSTM** | `GloVe` | **0.7044** | **0.7297** | 0.7283 | 0.6804 | 0.7428 |
| **11** | **Word2Vec + BiLSTM** | `Word2Vec` | **0.7037** | **0.7207** | 0.7236 | 0.6851 | 0.7307 |
| **12** | **FastText + BiLSTM** | `FastText` | **0.7024** | **0.7175** | 0.7196 | 0.6807 | 0.7471 |
| **13** | **GloVe + 2-Stacked BiLSTM** | `GloVe` | **0.6997** | **0.7245** | 0.7202 | 0.6788 | 0.7329 |
| **14** | **Word2Vec + BiRNN** | `Word2Vec` | **0.6995** | **0.7273** | 0.7250 | 0.6777 | 0.7337 |
| **15** | **Word2Vec + 2-Stacked BiRNN** | `Word2Vec` | **0.6985** | **0.7236** | 0.7234 | 0.6839 | 0.7199 |
| **16** | **Word2Vec + Simple RNN** | `Word2Vec` | **0.6979** | **0.7207** | 0.7177 | 0.6779 | 0.7306 |
| **17** | **FastText + BiRNN** | `FastText` | **0.6931** | **0.7123** | 0.7131 | 0.6676 | 0.7365 |
| **18** | **GloVe + BiRNN** | `GloVe` | **0.6900** | **0.7124** | 0.7095 | 0.6637 | 0.7397 |
| **19** | **Word2Vec + 2-Stacked BiLSTM** | `Word2Vec` | **0.6892** | **0.7146** | 0.7137 | 0.6666 | 0.7450 |
| **20** | **BoW + LinearSVC** | `Classical ML` | **0.6672** | **0.6916** | 0.6904 | 0.6647 | 0.6699 |
| **21** | **Word TF-IDF + MultinomialNB** | `Classical ML` | **0.6375** | **0.6904** | 0.6832 | 0.6769 | 0.6161 |
| **22** | **BoW + MultinomialNB** | `Classical ML` | **0.6348** | **0.6766** | 0.6728 | 0.6135 | 0.6731 |
| **23** | **Word TF-IDF + ComplementNB** | `Classical ML` | **0.6235** | **0.6966** | 0.6707 | 0.6597 | 0.6263 |
| **24** | **BoW + ComplementNB** | `Classical ML` | **0.6192** | **0.6945** | 0.6654 | 0.6616 | 0.6244 |

---
## Key Benchmark Insights:
1. **Top Transformer Performers:** **BERT Base Uncased** (Macro F1: `0.7540`, Accuracy: `0.7710`) and **RoBERTa Base** (Macro F1: `0.7529`, Accuracy: `0.7710`) achieved state-of-the-art performance, surpassing all traditional baselines by +3.7% in Macro F1 and +3.5% in Accuracy.
2. **Top Classical ML Model:** **Word TF-IDF + LogisticRegression** achieved Macro F1 `0.7172` and Accuracy `0.7363`, outperforming standard recurrent neural networks (RNN/LSTM/BiLSTM).
3. **Recurrent Embeddings Comparison:**
   - **Word2Vec + LSTM:** Macro F1 `0.7080` (Top among Word2Vec models)
   - **FastText + 2-Stacked BiLSTM:** Macro F1 `0.7060` (Top among FastText models)
   - **GloVe + BiLSTM:** Macro F1 `0.7044` (Top among GloVe models)
