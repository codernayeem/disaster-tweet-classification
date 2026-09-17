"""
Model 2: Word2Vec + Bidirectional RNN with Attention Pooling
Dataset: HumAID (QCRI/HumAID-all)
Author: NLP Project Team (Md. Nayeem & MD Jahid Hasan Jim)
"""

import html
import os
import random
import re
import time
from pathlib import Path
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Set seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def clean_tweet_text(text: str) -> str:
    """Cleans raw tweet text."""
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'@\w+', ' ', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'\brt\b', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def save_prediction_samples(file_path: Path, tweets, true_labels, pred_labels, confidences, max_samples=20):
    """Saves formatted prediction examples to a text file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        count = min(len(tweets), max_samples)
        f.write(f"{'='*80}\n")
        f.write(f" SAMPLE PREDICTIONS (Total Shown: {count})\n")
        f.write(f"{'='*80}\n\n")
        for i in range(count):
            f.write(f"Sample #{i + 1}\n")
            f.write(f"Tweet Text      : {tweets[i]}\n")
            f.write(f"Original Label  : {true_labels[i]}\n")
            f.write(f"Predicted Label : {pred_labels[i]}\n")
            f.write(f"Confidence      : {confidences[i]:.4f} ({confidences[i]*100:.2f}%)\n")
            f.write(f"{'-'*80}\n\n")


class AttentionPooling(nn.Module):
    """Computes learned attention weights over RNN time-steps."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, rnn_outputs):
        scores = self.attn(rnn_outputs)  # (batch_size, seq_len, 1)
        weights = F.softmax(scores, dim=1)
        context = torch.sum(weights * rnn_outputs, dim=1)  # (batch_size, hidden_dim)
        return context


class BiRNNClassifier(nn.Module):
    """Bidirectional RNN with pre-trained Word2Vec embeddings and Self-Attention Pooling."""

    def __init__(
        self,
        embedding_matrix,
        num_classes: int = 10,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        vocab_size, embed_dim = embedding_matrix.shape

        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float32),
            freeze=False,
            padding_idx=0
        )

        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            nonlinearity='tanh',
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.attention = AttentionPooling(hidden_dim * 2)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        embeds = self.embedding(x)
        rnn_out, _ = self.rnn(embeds)
        context = self.attention(rnn_out)
        normed = self.layer_norm(context)
        dropped = self.dropout(normed)
        logits = self.fc(dropped)
        return logits


class TweetSequenceDataset(Dataset):
    """PyTorch Dataset converting tokenized strings into fixed-length index tensors."""

    def __init__(self, texts, labels, word2idx, max_len=64):
        self.labels = np.array(labels)
        self.max_len = max_len
        self.sequences = []

        pad_idx = word2idx.get('<PAD>', 0)
        unk_idx = word2idx.get('<UNK>', 1)

        for text in texts:
            tokens = text.lower().split()
            seq = [word2idx.get(t, unk_idx) for t in tokens]
            if len(seq) < max_len:
                seq = seq + [pad_idx] * (max_len - len(seq))
            else:
                seq = seq[:max_len]
            self.sequences.append(seq)

        self.sequences = torch.tensor(self.sequences, dtype=torch.long)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def main():
    start_time = time.time()
    data_dir = Path("dataset")
    out_dir = Path("results/model_2")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 75)
    print(f"      MODEL 2: WORD2VEC + BIDIRECTIONAL RNN ({device.type.upper()} RUN)")
    print("=" * 75)

    # 1. Load Parquet Splits
    print("[+] Loading dataset from './dataset/'...")
    train_df = pd.read_parquet(data_dir / "train.parquet")
    val_df = pd.read_parquet(data_dir / "validation.parquet")
    test_df = pd.read_parquet(data_dir / "test.parquet")
    print(f"    - Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    # 2. Preprocess Text
    print("[+] Preprocessing tweet text...")
    train_df['clean_text'] = train_df['tweet_text'].apply(clean_tweet_text)
    val_df['clean_text'] = val_df['tweet_text'].apply(clean_tweet_text)
    test_df['clean_text'] = test_df['tweet_text'].apply(clean_tweet_text)

    # 3. Label Encoding
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df['class_label'])
    y_val = label_encoder.transform(val_df['class_label'])
    y_test = label_encoder.transform(test_df['class_label'])
    class_names = list(label_encoder.classes_)
    num_classes = len(class_names)
    print(f"[+] Encoded {num_classes} target classes.")

    # 4. Class Weights for Macro F1 Optimization
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.arange(num_classes),
        y=y_train
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # 5. Train Domain Word2Vec Skip-Gram Model
    EMBED_DIM = 128
    MAX_SEQ_LEN = 64
    all_texts = list(train_df['clean_text']) + list(val_df['clean_text']) + list(test_df['clean_text'])
    tokenized_corpus = [t.lower().split() for t in all_texts]

    print(f"[+] Training Skip-Gram Word2Vec (dim={EMBED_DIM}, window=5, min_count=2)...")
    w2v = Word2Vec(
        sentences=tokenized_corpus,
        vector_size=EMBED_DIM,
        window=5,
        min_count=2,
        sg=1,
        workers=4,
        seed=SEED,
        epochs=8
    )

    word2idx = {'<PAD>': 0, '<UNK>': 1}
    embedding_matrix = [np.zeros(EMBED_DIM), np.random.uniform(-0.1, 0.1, EMBED_DIM)]
    for word in w2v.wv.index_to_key:
        word2idx[word] = len(word2idx)
        embedding_matrix.append(w2v.wv[word])
    embedding_matrix = np.array(embedding_matrix, dtype=np.float32)
    print(f"[+] Vocabulary: {len(word2idx):,} words. Embedding Matrix: {embedding_matrix.shape}")

    # 6. DataLoaders
    BATCH_SIZE = 128
    train_ds = TweetSequenceDataset(train_df['clean_text'], y_train, word2idx, MAX_SEQ_LEN)
    val_ds = TweetSequenceDataset(val_df['clean_text'], y_val, word2idx, MAX_SEQ_LEN)
    test_ds = TweetSequenceDataset(test_df['clean_text'], y_test, word2idx, MAX_SEQ_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 7. Model Initialization
    model = BiRNNClassifier(
        embedding_matrix=embedding_matrix,
        num_classes=num_classes,
        hidden_dim=128,
        num_layers=2,
        dropout=0.3
    )

    model = train_torch_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights_tensor,
        max_epochs=15,
        patience=3,
        lr=1.5e-3,
        model_name="Model 2 (BiRNN)"
    )

    # 9. Test Split Evaluation with Softmax Confidences
    print("\n[+] Evaluating Model 2 on Test Split...")
    model.eval()
    test_preds, test_confidences = [], []
    with torch.no_grad():
        for seqs, _ in test_loader:
            seqs = seqs.to(device)
            logits = model(seqs)
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=-1)
            confs = np.max(probs, axis=-1)
            test_preds.extend(preds)
            test_confidences.extend(confs)

    y_test_pred = np.array(test_preds)
    confidences = np.array(test_confidences)

    test_acc = accuracy_score(y_test, y_test_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_test, y_test_pred, average='macro', zero_division=0)
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(y_test, y_test_pred, average='micro', zero_division=0)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(y_test, y_test_pred, average='weighted', zero_division=0)

    elapsed = time.time() - start_time
    print("\n" + "=" * 75)
    print("                     MODEL 2 TEST RESULTS")
    print("=" * 75)
    print(f"Accuracy          : {test_acc * 100:.2f}%")
    print(f"Macro F1-Score    : {macro_f1 * 100:.2f}%  <-- Primary Target Metric")
    print(f"Micro F1-Score    : {micro_f1 * 100:.2f}%")
    print(f"Weighted F1-Score : {weighted_f1 * 100:.2f}%")
    print(f"Macro Precision   : {macro_p * 100:.2f}%")
    print(f"Macro Recall      : {macro_r * 100:.2f}%")
    print(f"Total Time Taken  : {elapsed:.2f} seconds")
    print("-" * 75)

    # 10. Save Full Classification Report
    report_str = classification_report(y_test, y_test_pred, target_names=class_names, digits=4, zero_division=0)
    print("\nDetailed Per-Class Classification Report:")
    print(report_str)

    report_file = out_dir / "classification_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("      MODEL 2: WORD2VEC + BIDIRECTIONAL RNN REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Accuracy          : {test_acc * 100:.4f}%\n")
        f.write(f"Macro F1-Score    : {macro_f1 * 100:.4f}%\n")
        f.write(f"Micro F1-Score    : {micro_f1 * 100:.4f}%\n")
        f.write(f"Weighted F1-Score : {weighted_f1 * 100:.4f}%\n")
        f.write(f"Macro Precision   : {macro_p * 100:.4f}%\n")
        f.write(f"Macro Recall      : {macro_r * 100:.4f}%\n\n")
        f.write("Detailed Breakdown:\n")
        f.write(report_str)
    print(f"[+] Saved full classification report to: {report_file}")

    # 11. Plot & Save Confusion Matrix (Counts)
    cm_counts = confusion_matrix(y_test, y_test_pred)
    short_labels = [c.replace('_', ' ') for c in class_names]

    plt.figure(figsize=(11, 9))
    sns.heatmap(cm_counts, annot=True, fmt='d', cmap='Blues', xticklabels=short_labels, yticklabels=short_labels)
    plt.title(f"Model 2: Confusion Matrix (Counts) - Accuracy: {test_acc*100:.2f}% | Macro F1: {macro_f1*100:.2f}%", fontsize=12, fontweight='bold', pad=12)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.ylabel("True Label", fontsize=11)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    cm_counts_path = out_dir / "confusion_matrix_counts.png"
    plt.savefig(cm_counts_path, dpi=300)
    plt.close()
    print(f"[+] Saved Confusion Matrix (Counts) to: {cm_counts_path}")

    # 12. Plot & Save Confusion Matrix (Normalized)
    cm_norm = confusion_matrix(y_test, y_test_pred, normalize='true')
    plt.figure(figsize=(11, 9))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=short_labels, yticklabels=short_labels)
    plt.title(f"Model 2: Normalized Confusion Matrix - Macro F1: {macro_f1*100:.2f}%", fontsize=12, fontweight='bold', pad=12)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.ylabel("True Label", fontsize=11)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    cm_norm_path = out_dir / "confusion_matrix_normalized.png"
    plt.savefig(cm_norm_path, dpi=300)
    plt.close()
    print(f"[+] Saved Confusion Matrix (Normalized) to: {cm_norm_path}")

    # 13. Plot & Save Per-Class Precision, Recall, and F1-Score (3 separate plots)
    p_per_class, r_per_class, f1_per_class, _ = precision_recall_fscore_support(
        y_test, y_test_pred, average=None, zero_division=0
    )
    formatted_class_names = [c.replace('_', ' ').title() for c in class_names]

    def save_metric_bar_chart(values, metric_name, file_name, color):
        plt.figure(figsize=(12, 6))
        bars = plt.barh(formatted_class_names, values * 100, color=color, edgecolor='black', alpha=0.85)
        plt.title(f"Model 2: Per-Class {metric_name} (Macro Avg: {np.mean(values)*100:.2f}%)", fontsize=13, fontweight='bold', pad=12)
        plt.xlabel(f"{metric_name} (%)", fontsize=11)
        plt.ylabel("Humanitarian Category", fontsize=11)
        plt.xlim(0, 105)
        plt.gca().invert_yaxis()
        for bar in bars:
            w = bar.get_width()
            plt.text(w + 1, bar.get_y() + bar.get_height() / 2, f"{w:.2f}%", va='center', fontsize=9, fontweight='bold')
        plt.tight_layout()
        plot_path = out_dir / file_name
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"[+] Saved Per-Class {metric_name} plot to: {plot_path}")

    save_metric_bar_chart(p_per_class, "Precision", "per_class_precision.png", "#2980b9")
    save_metric_bar_chart(r_per_class, "Recall", "per_class_recall.png", "#27ae60")
    save_metric_bar_chart(f1_per_class, "F1-Score", "per_class_f1_score.png", "#e67e22")

    # 14. Extract Correct and Incorrect Predictions (Max 20 examples each)
    raw_tweets = test_df['tweet_text'].values
    true_label_names = np.array([class_names[i] for i in y_test])
    pred_label_names = np.array([class_names[i] for i in y_test_pred])

    correct_mask = (y_test == y_test_pred)
    incorrect_mask = (y_test != y_test_pred)

    correct_file = out_dir / "correct_predictions.txt"
    save_prediction_samples(
        correct_file,
        tweets=raw_tweets[correct_mask],
        true_labels=true_label_names[correct_mask],
        pred_labels=pred_label_names[correct_mask],
        confidences=confidences[correct_mask],
        max_samples=20
    )
    print(f"[+] Saved 20 correct prediction samples to: {correct_file}")

    incorrect_file = out_dir / "incorrect_predictions.txt"
    save_prediction_samples(
        incorrect_file,
        tweets=raw_tweets[incorrect_mask],
        true_labels=true_label_names[incorrect_mask],
        pred_labels=pred_label_names[incorrect_mask],
        confidences=confidences[incorrect_mask],
        max_samples=20
    )
    print(f"[+] Saved 20 incorrect prediction samples to: {incorrect_file}")

    # 15. Save Metrics Summary CSV
    metrics_df = pd.DataFrame([{
        'Model': 'Word2Vec + BiRNN',
        'Accuracy': test_acc,
        'Macro_F1': macro_f1,
        'Micro_F1': micro_f1,
        'Weighted_F1': weighted_f1,
        'Macro_Precision': macro_p,
        'Macro_Recall': macro_r,
        'Elapsed_Seconds': elapsed
    }])
    metrics_df.to_csv(out_dir / "metrics.csv", index=False)
    print(f"\n[SUCCESS] All Model 2 artifacts saved successfully to '{out_dir}/'!")
    print("=" * 75)


if __name__ == '__main__':
    main()
