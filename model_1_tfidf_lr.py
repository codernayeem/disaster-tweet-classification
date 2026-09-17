"""
Model 1: Sublinear TF-IDF + Balanced Logistic Regression
Dataset: HumAID (QCRI/HumAID-all)
Author: NLP Project Team (Md. Nayeem & MD Jahid Hasan Jim)
"""

import html
import os
import re
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import LabelEncoder


# --------------------------------------------------------------------------
# Contraction, Slang & Emoji Dictionaries
# --------------------------------------------------------------------------
CONTRACTIONS = {
    "can't": "cannot", "won't": "will not", "n't": " not", "'re": " are",
    "'s": " is", "'d": " would", "'ll": " will", "'t": " not", "'ve": " have",
    "'m": " am", "i'm": "i am", "we're": "we are", "they're": "they are",
    "it's": "it is", "there's": "there is", "that's": "that is", "what's": "what is",
    "here's": "here is", "let's": "let us", "who's": "who is", "how's": "how is"
}

DISASTER_SLANG = {
    r"\bpls\b": "please", r"\bplz\b": "please", r"\bthx\b": "thanks",
    r"\bu\b": "you", r"\bur\b": "your", r"\br\b": "are",
    r"\bw/\b": "with", r"\bw/o\b": "without", r"\bb4\b": "before",
    r"\bmsg\b": "message", r"\binfo\b": "information", r"\bemerg\b": "emergency",
    r"\bevac\b": "evacuation", r"\bevacs\b": "evacuations", r"\bvicts\b": "victims",
    r"\bgov\b": "government", r"\bdept\b": "department", r"\bvol\b": "volunteer"
}

EMOJI_TRANSLATIONS = {
    "🙏": " prayer support ", "💔": " heartbreak grief ", "❤️": " love sympathy ",
    "🚨": " emergency warning alert ", "⚠️": " danger warning caution ",
    "🔥": " fire wildfire disaster ", "🌊": " flood tsunami water surge ",
    "🌧️": " rain storm hurricane ", "🌪️": " tornado storm ", "⚡": " storm lightning ",
    "😢": " crying sorrow sadness ", "😭": " weeping tragedy ", "🕯️": " mourning memorial ",
    "🆘": " urgent help request emergency ", "🏠": " shelter home house ", "🏥": " hospital medical clinic "
}


def split_camel_case(text: str) -> str:
    """Splits CamelCase words in hashtags (#HurricaneHarvey -> Hurricane Harvey)."""
    return re.sub(r'([a-z])([A-Z])', r'\1 \2', text)


def clean_tweet_text(text: str) -> str:
    """Comprehensive disaster tweet cleaner."""
    if not isinstance(text, str):
        return ""
    # 1. HTML Unescape
    text = html.unescape(text)
    # 2. Emoji translation into semantic text tokens
    for em, rep in EMOJI_TRANSLATIONS.items():
        text = text.replace(em, rep)
    # 3. Strip URLs and @mentions
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'@\w+', ' ', text)
    # 4. Split CamelCase hashtags
    text = re.sub(r'#(\w+)', lambda m: split_camel_case(m.group(1)), text)
    # 5. Lowercase and expand contractions
    text = text.lower()
    for c, exp in CONTRACTIONS.items():
        text = text.replace(c, exp)
    # 6. Normalize disaster abbreviations and slang
    for pattern, rep in DISASTER_SLANG.items():
        text = re.sub(pattern, rep, text, flags=re.IGNORECASE)
    # 7. Reduce repeated character elongations (sooooo -> soo)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    # 8. Remove RT tokens & standardize special quotes
    text = re.sub(r'\brt\b', ' ', text, flags=re.IGNORECASE)
    text = text.replace('"', '"').replace('"', '"').replace(''', "'").replace(''', "'")
    # 9. Clean multi-spaces
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


def main():
    start_time = time.time()
    data_dir = Path("dataset")
    out_dir = Path("results/model_1")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("      MODEL 1: TF-IDF + BALANCED LOGISTIC REGRESSION")
    print("=" * 75)

    # 1. Load Parquet Splits
    print("[+] Loading dataset from './dataset/'...")
    train_df = pd.read_parquet(data_dir / "train.parquet")
    val_df = pd.read_parquet(data_dir / "validation.parquet")
    test_df = pd.read_parquet(data_dir / "test.parquet")
    print(f"    - Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    # 2. Preprocess Text
    print("[+] Preprocessing tweet text with comprehensive disaster NLP cleaner...")
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

    # 4. Feature Extraction: Sublinear TF-IDF (1-2 ngrams, 20,000 features)
    print("[+] Extracting TF-IDF features (unigrams + bigrams, sublinear_tf=True, max_features=20,000)...")
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=20000,
        sublinear_tf=True,
        strip_accents='unicode'
    )
    X_train = tfidf.fit_transform(train_df['clean_text'])
    X_val = tfidf.transform(val_df['clean_text'])
    X_test = tfidf.transform(test_df['clean_text'])
    print(f"    - TF-IDF Matrix shape: Train={X_train.shape}, Test={X_test.shape}")

    # 5. Model Training: Logistic Regression with Balanced Class Weights (C=2.0)
    print("[+] Training Logistic Regression with class_weight='balanced' (C=2.0)...")
    clf = LogisticRegression(
        C=2.0,
        max_iter=1000,
        class_weight='balanced',
        random_state=42,
        solver='lbfgs',
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    # 6. Evaluation on Validation Set
    y_val_pred = clf.predict(X_val)
    _, _, val_macro_f1, _ = precision_recall_fscore_support(y_val, y_val_pred, average='macro', zero_division=0)
    print(f"[+] Validation Macro F1-Score: {val_macro_f1 * 100:.2f}%")

    # 7. Evaluation on Test Set with Probabilities & Confidences
    print("[+] Evaluating Model 1 on Test Split...")
    probs = clf.predict_proba(X_test)
    y_test_pred = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)

    test_acc = accuracy_score(y_test, y_test_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_test, y_test_pred, average='macro', zero_division=0)
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(y_test, y_test_pred, average='micro', zero_division=0)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(y_test, y_test_pred, average='weighted', zero_division=0)

    elapsed = time.time() - start_time
    print("\n" + "=" * 75)
    print("                     MODEL 1 TEST RESULTS")
    print("=" * 75)
    print(f"Accuracy          : {test_acc * 100:.2f}%")
    print(f"Macro F1-Score    : {macro_f1 * 100:.2f}%  <-- Primary Target Metric")
    print(f"Micro F1-Score    : {micro_f1 * 100:.2f}%")
    print(f"Weighted F1-Score : {weighted_f1 * 100:.2f}%")
    print(f"Macro Precision   : {macro_p * 100:.2f}%")
    print(f"Macro Recall      : {macro_r * 100:.2f}%")
    print(f"Total Time Taken  : {elapsed:.2f} seconds")
    print("-" * 75)

    # 8. Save Full Classification Report
    report_str = classification_report(y_test, y_test_pred, target_names=class_names, digits=4, zero_division=0)
    print("\nDetailed Per-Class Classification Report:")
    print(report_str)

    report_file = out_dir / "classification_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("      MODEL 1: TF-IDF + BALANCED LOGISTIC REGRESSION REPORT\n")
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

    # 9. Plot & Save Confusion Matrix (Counts)
    cm_counts = confusion_matrix(y_test, y_test_pred)
    short_labels = [c.replace('_', ' ') for c in class_names]

    plt.figure(figsize=(11, 9))
    sns.heatmap(cm_counts, annot=True, fmt='d', cmap='Blues', xticklabels=short_labels, yticklabels=short_labels)
    plt.title(f"Model 1: Confusion Matrix (Counts) - Accuracy: {test_acc*100:.2f}% | Macro F1: {macro_f1*100:.2f}%", fontsize=12, fontweight='bold', pad=12)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.ylabel("True Label", fontsize=11)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    cm_counts_path = out_dir / "confusion_matrix_counts.png"
    plt.savefig(cm_counts_path, dpi=300)
    plt.close()
    print(f"[+] Saved Confusion Matrix (Counts) to: {cm_counts_path}")

    # 10. Plot & Save Confusion Matrix (Normalized)
    cm_norm = confusion_matrix(y_test, y_test_pred, normalize='true')
    plt.figure(figsize=(11, 9))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=short_labels, yticklabels=short_labels)
    plt.title(f"Model 1: Normalized Confusion Matrix - Macro F1: {macro_f1*100:.2f}%", fontsize=12, fontweight='bold', pad=12)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.ylabel("True Label", fontsize=11)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    cm_norm_path = out_dir / "confusion_matrix_normalized.png"
    plt.savefig(cm_norm_path, dpi=300)
    plt.close()
    print(f"[+] Saved Confusion Matrix (Normalized) to: {cm_norm_path}")

    # 11. Plot & Save Per-Class Precision, Recall, and F1-Score (3 separate plots)
    p_per_class, r_per_class, f1_per_class, _ = precision_recall_fscore_support(
        y_test, y_test_pred, average=None, zero_division=0
    )
    formatted_class_names = [c.replace('_', ' ').title() for c in class_names]

    def save_metric_bar_chart(values, metric_name, file_name, color):
        plt.figure(figsize=(12, 6))
        bars = plt.barh(formatted_class_names, values * 100, color=color, edgecolor='black', alpha=0.85)
        plt.title(f"Model 1: Per-Class {metric_name} (Macro Avg: {np.mean(values)*100:.2f}%)", fontsize=13, fontweight='bold', pad=12)
        plt.xlabel(f"{metric_name} (%)", fontsize=11)
        plt.ylabel("Humanitarian Category", fontsize=11)
        plt.xlim(0, 105)
        plt.gca().invert_yaxis()  # top-down order
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

    # 12. Extract Correct and Incorrect Predictions (Max 20 examples each)
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

    # 13. Save Metrics Summary CSV
    metrics_df = pd.DataFrame([{
        'Model': 'TF-IDF + Logistic Regression',
        'Accuracy': test_acc,
        'Macro_F1': macro_f1,
        'Micro_F1': micro_f1,
        'Weighted_F1': weighted_f1,
        'Macro_Precision': macro_p,
        'Macro_Recall': macro_r,
        'Elapsed_Seconds': elapsed
    }])
    metrics_df.to_csv(out_dir / "metrics.csv", index=False)
    print(f"\n[SUCCESS] All Model 1 artifacts saved successfully to '{out_dir}/'!")
    print("=" * 75)


if __name__ == '__main__':
    main()
