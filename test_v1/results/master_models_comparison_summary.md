# Comprehensive Disaster Tweet Classification Models Benchmark Summary

**Generated on:** 2026-09-19 16:47:45
**Total Models Evaluated:** 15 / 15

### Global Leaderboard (Ranked by Macro F1 Score)

| Rank | Model Name | Architecture Family | Accuracy | Macro F1 | Weighted F1 | Folder |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| 1 | **BERT (bert-base-uncased)** | Transformer (Fine-tuned) | 75.33% | **73.18%** | 74.94% | [`13_model_12_bert_classifier`](file:///d:/nlp-project/results/13_model_12_bert_classifier) |
| 2 | **RoBERTa (roberta-base)** | Transformer (Fine-tuned) | 74.54% | **72.90%** | 74.36% | [`14_model_13_roberta_classifier`](file:///d:/nlp-project/results/14_model_13_roberta_classifier) |
| 3 | **DeBERTa-v3 (deberta-v3-base)** | Transformer (Fine-tuned) | 73.93% | **72.00%** | 72.41% | [`15_model_14_deberta_classifier`](file:///d:/nlp-project/results/15_model_14_deberta_classifier) |
| 4 | **Model 1 (TF-IDF + LR)** | Classical ML (TF-IDF) | 73.51% | **71.53%** | 73.76% | [`01_model_1_tfidf_logistic_regression`](file:///d:/nlp-project/results/01_model_1_tfidf_logistic_regression) |
| 5 | **Preprocessing Tier 3 (Standard Clean)** | Classical ML (TF-IDF) | 73.51% | **71.53%** | 73.76% | [`02_model_4_tfidf_preprocessing_ablation`](file:///d:/nlp-project/results/02_model_4_tfidf_preprocessing_ablation) |
| 6 | **Model 5 (Char TF-IDF + LR)** | Classical ML (TF-IDF) | 73.06% | **70.68%** | 73.41% | [`03_model_5_tfidf_char_logistic_regression`](file:///d:/nlp-project/results/03_model_5_tfidf_char_logistic_regression) |
| 7 | **Model 6 (TF-IDF + LinearSVM)** | Classical ML (TF-IDF) | 73.07% | **70.65%** | 72.84% | [`04_model_6_tfidf_linear_svm`](file:///d:/nlp-project/results/04_model_6_tfidf_linear_svm) |
| 8 | **Model 10 (Stacked BiLSTM)** | Word2Vec + Deep RNN/LSTM | 70.82% | **68.23%** | 70.81% | [`11_model_10_word2vec_stacked_bilstm`](file:///d:/nlp-project/results/11_model_10_word2vec_stacked_bilstm) |
| 9 | **Model 3 (BiLSTM Baseline)** | Word2Vec + Deep RNN/LSTM | 69.51% | **67.85%** | 69.69% | [`10_model_3_word2vec_bilstm`](file:///d:/nlp-project/results/10_model_3_word2vec_bilstm) |
| 10 | **Model 11 (BiLSTM + Attention)** | Word2Vec + Deep RNN/LSTM | 70.33% | **67.82%** | 69.88% | [`12_model_11_word2vec_bilstm_attention`](file:///d:/nlp-project/results/12_model_11_word2vec_bilstm_attention) |
| 11 | **Model 9 (Stacked BiRNN)** | Word2Vec + Deep RNN/LSTM | 69.74% | **67.17%** | 69.92% | [`08_model_9_word2vec_stacked_birnn`](file:///d:/nlp-project/results/08_model_9_word2vec_stacked_birnn) |
| 12 | **Model 2 (BiRNN + Attention)** | Word2Vec + Deep RNN/LSTM | 68.63% | **66.48%** | 68.87% | [`06_model_2_word2vec_birnn`](file:///d:/nlp-project/results/06_model_2_word2vec_birnn) |
| 13 | **Model (Unidirectional LSTM)** | Word2Vec + Deep RNN/LSTM | 69.37% | **65.11%** | 69.95% | [`09_model_word2vec_lstm`](file:///d:/nlp-project/results/09_model_word2vec_lstm) |
| 14 | **Model 7 (Simple RNN)** | Word2Vec + Deep RNN/LSTM | 9.56% | **1.83%** | 1.76% | [`05_model_7_word2vec_simple_rnn`](file:///d:/nlp-project/results/05_model_7_word2vec_simple_rnn) |
| 15 | **Model 8 (Stacked RNN)** | Word2Vec + Deep RNN/LSTM | 9.54% | **1.74%** | 1.66% | [`07_model_8_word2vec_stacked_rnn`](file:///d:/nlp-project/results/07_model_8_word2vec_stacked_rnn) |

