"""
Streamlit Web Application: Disaster Tweet Classification Benchmark
HumAID Dataset (76,484 Tweets across 10 Humanitarian Target Classes)
Robust Multi-Architecture Inference Engine (Classical ML, Word2Vec, FastText, GloVe, BERT, RoBERTa)
"""

import os
import re
import html
import json
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import torch
import torch.nn as nn

# --------------------------------------------------------------------------
# Page Configuration & Modern Design System
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Disaster Tweet Classification | AI Crisis Response",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 24px 30px;
        border-radius: 12px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    .main-title {
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #f8fafc;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .main-subtitle {
        font-size: 0.95rem;
        color: #94a3b8;
        margin-top: 6px;
        margin-bottom: 0;
    }
    
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    
    .pred-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 20px 24px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    
    .badge-rank {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .rank-1 { background: #fef3c7; color: #b45309; border: 1px solid #fde68a; }
    .rank-2 { background: #e0e7ff; color: #4338ca; border: 1px solid #c7d2fe; }
    .rank-3 { background: #ecfdf5; color: #047857; border: 1px solid #a7f3d0; }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 18px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Paths & Constant Metadata
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
DATASET_DIR = BASE_DIR / "dataset"

CLASS_INFO = {
    "caution_and_advice": {"label": "Caution & Advice", "color": "#d97706", "icon": "⚠️"},
    "displaced_people_and_evacuations": {"label": "Displaced People & Evacuations", "color": "#7c3aed", "icon": "🚶‍♂️"},
    "infrastructure_and_utility_damage": {"label": "Infrastructure & Utility Damage", "color": "#475569", "icon": "🏗️"},
    "injured_or_dead_people": {"label": "Injured or Dead People", "color": "#dc2626", "icon": "🚑"},
    "missing_or_found_people": {"label": "Missing or Found People", "color": "#ea580c", "icon": "🔍"},
    "not_humanitarian": {"label": "Not Humanitarian", "color": "#64748b", "icon": "💬"},
    "other_relevant_information": {"label": "Other Relevant Information", "color": "#0284c7", "icon": "ℹ️"},
    "requests_or_urgent_needs": {"label": "Requests & Urgent Needs", "color": "#b91c1c", "icon": "🆘"},
    "rescue_volunteering_or_donation_effort": {"label": "Rescue, Volunteering & Donations", "color": "#16a34a", "icon": "🤝"},
    "sympathy_and_support": {"label": "Sympathy & Support", "color": "#db2777", "icon": "❤️"}
}

CLASS_NAMES = sorted(list(CLASS_INFO.keys()))

# --------------------------------------------------------------------------
# Preprocessing Engine
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
    return re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

def clean_tweet_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    for em, rep in EMOJI_TRANSLATIONS.items():
        text = text.replace(em, rep)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'@\w+', ' ', text)
    text = re.sub(r'#(\w+)', lambda m: split_camel_case(m.group(1)), text)
    text = text.lower()
    for c, exp in CONTRACTIONS.items():
        text = text.replace(c, exp)
    for pattern, rep in DISASTER_SLANG.items():
        text = re.sub(pattern, rep, text, flags=re.IGNORECASE)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'\brt\b', ' ', text, flags=re.IGNORECASE)
    text = text.replace('"', '"').replace('"', '"').replace('’', "'").replace('‘', "'")
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --------------------------------------------------------------------------
# PyTorch Recurrent Architecture Definition
# --------------------------------------------------------------------------
class RecurrentClassifier(nn.Module):
    def __init__(self, cell_type: str, vocab_size: int, embed_dim: int, hidden_dim: int = 128, 
                 num_layers: int = 1, dropout: float = 0.3, num_classes: int = 10):
        super().__init__()
        self.cell_type = cell_type.lower()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        rnn_dropout = dropout if num_layers > 1 else 0.0
        if self.cell_type == 'rnn':
            self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True, nonlinearity='tanh', dropout=rnn_dropout)
        elif self.cell_type == 'lstm':
            self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=rnn_dropout)
        elif self.cell_type == 'gru':
            self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=rnn_dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.rnn(emb)
        pooled = torch.max(out, dim=1)[0]
        pooled = self.dropout(pooled)
        return self.fc(pooled)

# --------------------------------------------------------------------------
# Dynamic Model Discovery & Loader Registry
# --------------------------------------------------------------------------
def discover_available_models():
    """
    Scans the results directory and dynamically returns only models that exist on disk.
    Never crashes even if models are missing.
    """
    catalog = {}
    
    # 1. Classical ML Models
    classical_saved = RESULTS_DIR / "01_classical_ml_tfidf_bow" / "saved_models"
    if classical_saved.exists():
        c_models = [
            ("Word TF-IDF + Logistic Regression", "word_tf_idf_plus_logisticregression", "classical_sklearn", 0.7241, 0.7433),
            ("Word TF-IDF + LinearSVC", "word_tf_idf_plus_linearsvc", "classical_sklearn_decision", 0.7233, 0.7485),
            ("BoW + LinearSVC", "bow_plus_linearsvc", "classical_sklearn_decision", 0.7180, 0.7404),
            ("BoW + Logistic Regression", "bow_plus_logisticregression", "classical_sklearn", 0.7161, 0.7355),
            ("Word TF-IDF + MultinomialNB", "word_tf_idf_plus_multinomialnb", "classical_sklearn", 0.6401, 0.6883),
            ("BoW + MultinomialNB", "bow_plus_multinomialnb", "classical_sklearn", 0.6309, 0.6793),
        ]
        for name, prefix, mtype, f1, acc in c_models:
            m_path = classical_saved / f"{prefix}.joblib"
            v_path = classical_saved / f"{prefix}_vectorizer.joblib"
            if m_path.exists() and v_path.exists():
                catalog[name] = {
                    "type": mtype, "model_path": m_path, "vec_path": v_path,
                    "macro_f1": f1, "accuracy": acc, "category": "Classical ML", "dim": "N-gram Features"
                }

    # 2. FastText Recurrent Models
    ft_base = RESULTS_DIR / "03_fasttext_recurrent_models"
    ft_vocab_path = ft_base / "fasttext" / "vocab2id.json"
    if ft_vocab_path.exists():
        ft_candidates = [
            ("FastText + BiLSTM", "bilstm", "lstm", 100, 256, 1, 0.7404, 0.7627),
            ("FastText + BiRNN", "birnn", "rnn", 100, 256, 1, 0.7258, 0.7421)
        ]
        for name, sub, cell, emb_dim, h_dim, n_lay, f1, acc in ft_candidates:
            p_path = ft_base / "models" / sub / "best_model.pt"
            if p_path.exists():
                catalog[name] = {
                    "type": "pytorch_recurrent", "model_path": p_path, "vocab_path": ft_vocab_path,
                    "cell_type": cell, "embed_dim": emb_dim, "hidden_dim": h_dim, "num_layers": n_lay,
                    "macro_f1": f1, "accuracy": acc, "category": "FastText Subword", "dim": f"{emb_dim}d"
                }

    # 3. Word2Vec Recurrent Models
    w2v_base = RESULTS_DIR / "02_word2vec_recurrent_models"
    w2v_vocab_path = w2v_base / "word2vec" / "vocab2id.json"
    if w2v_vocab_path.exists():
        w2v_candidates = [
            ("Word2Vec + 2-Stacked BiLSTM", "stacked_bilstm", "lstm", 300, 256, 2, 0.7369, 0.7549),
            ("Word2Vec + BiLSTM", "bilstm", "lstm", 300, 256, 1, 0.7329, 0.7567),
            ("Word2Vec + BiGRU", "bigru", "gru", 300, 128, 1, 0.7279, 0.7516),
            ("Word2Vec + BiRNN", "birnn", "rnn", 300, 64, 1, 0.7090, 0.7406),
        ]
        for name, sub, cell, emb_dim, h_dim, n_lay, f1, acc in w2v_candidates:
            p_path = w2v_base / "models" / sub / "best_model.pt"
            if p_path.exists():
                catalog[name] = {
                    "type": "pytorch_recurrent", "model_path": p_path, "vocab_path": w2v_vocab_path,
                    "cell_type": cell, "embed_dim": emb_dim, "hidden_dim": h_dim, "num_layers": n_lay,
                    "macro_f1": f1, "accuracy": acc, "category": "Word2Vec", "dim": f"{emb_dim}d"
                }

    # 4. GloVe Recurrent Models
    glove_base = RESULTS_DIR / "04_glove_recurrent_models"
    glove_vocab_path = glove_base / "glove" / "vocab2id.json"
    if glove_vocab_path.exists():
        glove_candidates = [
            ("GloVe + BiLSTM", "bilstm", "lstm", 100, 256, 1, 0.7350, 0.7580),
            ("GloVe + BiRNN", "birnn", "rnn", 100, 128, 1, 0.7180, 0.7410),
        ]
        for name, sub, cell, emb_dim, h_dim, n_lay, f1, acc in glove_candidates:
            p_path = glove_base / "models" / sub / "best_model.pt"
            if p_path.exists():
                catalog[name] = {
                    "type": "pytorch_recurrent", "model_path": p_path, "vocab_path": glove_vocab_path,
                    "cell_type": cell, "embed_dim": emb_dim, "hidden_dim": h_dim, "num_layers": n_lay,
                    "macro_f1": f1, "accuracy": acc, "category": "GloVe Pretrained", "dim": f"{emb_dim}d"
                }

    # 5. Transformer Models (BERT & RoBERTa)
    bert_path = RESULTS_DIR / "05_transformer_bert" / "models" / "bert_base_uncased" / "best_model.pt"
    if bert_path.exists():
        catalog["BERT Base Uncased"] = {
            "type": "transformer", "model_path": bert_path, "hf_id": "bert-base-uncased",
            "macro_f1": 0.7850, "accuracy": 0.8020, "category": "Transformer (BERT)", "dim": "768d"
        }
        
    roberta_path = RESULTS_DIR / "06_transformer_roberta" / "models" / "roberta_base" / "best_model.pt"
    if roberta_path.exists():
        catalog["RoBERTa Base"] = {
            "type": "transformer", "model_path": roberta_path, "hf_id": "roberta-base",
            "macro_f1": 0.7910, "accuracy": 0.8090, "category": "Transformer (RoBERTa)", "dim": "768d"
        }

    return catalog

AVAILABLE_MODELS = discover_available_models()

@st.cache_resource
def load_model_bundle(model_name: str):
    if model_name not in AVAILABLE_MODELS:
        return None
        
    cfg = AVAILABLE_MODELS[model_name]
    m_type = cfg["type"]
    
    try:
        if m_type in ["classical_sklearn", "classical_sklearn_decision"]:
            vec = joblib.load(cfg["vec_path"])
            clf = joblib.load(cfg["model_path"])
            return {"type": m_type, "vec": vec, "clf": clf}
            
        elif m_type == "pytorch_recurrent":
            with open(cfg["vocab_path"], "r", encoding="utf-8") as f:
                vocab = json.load(f)
            model = RecurrentClassifier(
                cell_type=cfg["cell_type"],
                vocab_size=len(vocab),
                embed_dim=cfg["embed_dim"],
                hidden_dim=cfg["hidden_dim"],
                num_layers=cfg["num_layers"],
                num_classes=len(CLASS_NAMES)
            )
            state_dict = torch.load(cfg["model_path"], map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
            model.eval()
            return {"type": "pytorch_recurrent", "model": model, "vocab": vocab, "max_len": 48}
            
        elif m_type == "transformer":
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            tok = AutoTokenizer.from_pretrained(cfg["hf_id"])
            model = AutoModelForSequenceClassification.from_pretrained(cfg["hf_id"], num_labels=len(CLASS_NAMES))
            state_dict = torch.load(cfg["model_path"], map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
            model.eval()
            return {"type": "transformer", "tok": tok, "model": model, "max_len": 48}
            
    except Exception as e:
        st.error(f"Error loading {model_name}: {e}")
        return None

def predict_single(model_bundle, raw_text: str):
    cleaned = clean_tweet_text(raw_text)
    if not cleaned or model_bundle is None:
        return None, 0.0, np.zeros(len(CLASS_NAMES)), cleaned
        
    m_type = model_bundle["type"]
    
    if m_type == "classical_sklearn":
        vec = model_bundle["vec"]
        clf = model_bundle["clf"]
        x = vec.transform([cleaned])
        probs = clf.predict_proba(x)[0]
        
    elif m_type == "classical_sklearn_decision":
        vec = model_bundle["vec"]
        clf = model_bundle["clf"]
        x = vec.transform([cleaned])
        df_scores = clf.decision_function(x)[0]
        exp_s = np.exp(df_scores - np.max(df_scores))
        probs = exp_s / exp_s.sum()
        
    elif m_type == "pytorch_recurrent":
        model = model_bundle["model"]
        vocab = model_bundle["vocab"]
        max_len = model_bundle.get("max_len", 48)
        tokens = cleaned.split()
        seq = [vocab.get(t, 1) for t in tokens][:max_len]
        if len(seq) < max_len:
            seq += [0] * (max_len - len(seq))
        x_tensor = torch.tensor([seq], dtype=torch.long)
        with torch.no_grad():
            logits = model(x_tensor)
            probs = torch.softmax(logits, dim=1).numpy()[0]
            
    elif m_type == "transformer":
        tok = model_bundle["tok"]
        model = model_bundle["model"]
        inputs = tok([cleaned], padding=True, truncation=True, max_length=model_bundle.get("max_len", 48), return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1).numpy()[0]
            
    pred_idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])
    return pred_class, confidence, probs, cleaned

# --------------------------------------------------------------------------
# Sidebar Navigation & Model Selector
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🚨 Crisis NLP System")
    st.caption("Humanitarian Disaster Tweet Classification")
    st.markdown("---")
    
    st.markdown("#### 🎯 Active Inference Model")
    
    if AVAILABLE_MODELS:
        # Sort available models by Macro F1
        sorted_model_names = sorted(AVAILABLE_MODELS.keys(), key=lambda k: AVAILABLE_MODELS[k]["macro_f1"], reverse=True)
        selected_model_name = st.selectbox(
            f"Select Model ({len(AVAILABLE_MODELS)} Available on Disk):",
            sorted_model_names,
            index=0,
            help="Select any trained model for live classification and diagnostic comparison."
        )
        active_cfg = AVAILABLE_MODELS[selected_model_name]
        
        st.markdown(f"""
        <div style="background:#f8fafc; padding:12px; border-radius:8px; border:1px solid #e2e8f0; margin-top:8px;">
            <div style="font-size:0.8rem; color:#64748b; font-weight:600;">FAMILY: <span style="color:#0f172a;">{active_cfg['category']}</span></div>
            <div style="font-size:0.95rem; font-weight:700; color:#0f172a; margin-top:2px;">{selected_model_name}</div>
            <div style="display:flex; justify-content:space-between; margin-top:10px;">
                <div>
                    <div style="font-size:0.75rem; color:#64748b;">Macro F1</div>
                    <div style="font-size:1.05rem; font-weight:700; color:#1f77b4;">{active_cfg['macro_f1']*100:.2f}%</div>
                </div>
                <div>
                    <div style="font-size:0.75rem; color:#64748b;">Accuracy</div>
                    <div style="font-size:1.05rem; font-weight:700; color:#ff7f0e;">{active_cfg['accuracy']*100:.2f}%</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        model_bundle = load_model_bundle(selected_model_name)
    else:
        st.warning("⚠️ No serialized models detected in results folder yet.")
        selected_model_name = None
        active_cfg = None
        model_bundle = None

    st.markdown("---")
    st.markdown("#### 📚 Benchmark Corpus")
    st.markdown("- **HumAID Corpus**: 76,484 tweets")
    st.markdown("- **Test Split**: 15,160 tweets")
    st.markdown("- **Classes**: 10 Crisis Categories")
    st.markdown(f"- **Models Available**: **{len(AVAILABLE_MODELS)} Loaded**")
    
    st.markdown("---")
    st.caption("CSE 4122 NLP Lab Project | HumAID Disaster Benchmark")

# --------------------------------------------------------------------------
# Main App Header
# --------------------------------------------------------------------------
st.markdown(f"""
<div class="main-header">
    <div class="main-title">
        <span>🚨 Disaster Tweet Classification System</span>
    </div>
    <div class="main-subtitle">
        Multi-architecture NLP benchmark & real-time crisis response classifier evaluated across 10 humanitarian crisis categories.
    </div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Tabs Architecture
# --------------------------------------------------------------------------
tab_classifier, tab_batch, tab_leaderboard, tab_dataset = st.tabs([
    "🔮 Live Classifier",
    "⚡ Batch Testing",
    "📊 Benchmark Leaderboard",
    "📖 Dataset & Preprocessing"
])

# ==========================================================================
# TAB 1: LIVE CLASSIFIER
# ==========================================================================
with tab_classifier:
    st.markdown("### 💬 Single Tweet Classification & Multi-Class Confidence")
    if selected_model_name:
        st.caption(f"Active Inference Model: **{selected_model_name}** ({active_cfg['category']})")
    
    PRESETS = {
        "Custom Input": "",
        "Urgent Request (SOS)": "SOS! We are trapped on the second floor on Pine St due to rapid flood waters. Need clean drinking water and baby formula urgently #HurricaneHarvey",
        "Missing Person Alert": "MISSING: 7-year-old Lucas Vance last seen wearing blue raincoat near Spring Creek after the flood. Contact emergency dispatch #MissingPerson",
        "Evacuation & Shelter": "Over 2,500 residents evacuated from Riverside community are now housed at the high school gymnasium shelter #FloodEvacuation",
        "Infrastructure Damage": "Power grid failed across 4 districts. Main bridge on Highway 10 is cracked and impassable due to landslides #EarthquakeDamage",
        "Caution & Advice": "FLASH FLOOD WARNING: Move to higher ground immediately. Do not drive through flooded roads. Follow local emergency broadcasts.",
        "Injuries & Casualties": "Officials report at least 14 injured and 3 dead following building collapse after 6.8 magnitude earthquake.",
        "Rescue & Volunteering": "Red Cross boats and volunteer crews delivered 500 meals and hygiene kits to cut-off neighborhoods today #DisasterRelief",
        "Sympathy & Support": "Sending prayers and heartfelt condolences to all families affected by the storm in Florida. Stay strong.",
        "Not Humanitarian (Casual)": "Having pizza while watching the new movie on Netflix tonight with friends."
    }
    
    col_preset, col_info = st.columns([2, 1])
    with col_preset:
        preset_choice = st.selectbox("Load Example Crisis Tweet:", list(PRESETS.keys()))
        default_text = PRESETS[preset_choice]
        
    user_input = st.text_area(
        "Enter or Paste Tweet Content:",
        value=default_text,
        height=100,
        placeholder="Type a tweet or crisis message here..."
    )
    
    col_c1, col_c2 = st.columns([1, 4])
    with col_c1:
        classify_btn = st.button("🚀 Classify Tweet", type="primary", use_container_width=True)
    with col_c2:
        if user_input:
            st.caption(f"Input: **{len(user_input)}** characters | **{len(user_input.split())}** words")
            
    if (classify_btn or user_input.strip()) and user_input.strip() and model_bundle:
        with st.spinner("Classifying tweet..."):
            pred_class, conf, probs, cleaned = predict_single(model_bundle, user_input)
            
        if pred_class:
            c_info = CLASS_INFO.get(pred_class, {"label": pred_class, "color": "#0f172a", "icon": "📌"})
            
            st.markdown("---")
            col_res, col_chart = st.columns([1, 1.3])
            
            with col_res:
                st.markdown(f"""
                <div class="pred-card" style="border-left: 6px solid {c_info['color']};">
                    <div style="font-size:0.85rem; font-weight:600; color:#64748b; text-transform:uppercase;">Predicted Crisis Category</div>
                    <div style="font-size:1.4rem; font-weight:800; color:{c_info['color']}; margin-top:4px;">
                        {c_info['icon']} {c_info['label']}
                    </div>
                    <div style="margin-top:16px;">
                        <div style="font-size:0.85rem; color:#64748b; font-weight:600;">Prediction Confidence</div>
                        <div style="font-size:2rem; font-weight:800; color:#0f172a;">{conf * 100:.2f}%</div>
                    </div>
                    <div style="margin-top:12px; font-size:0.8rem; color:#64748b;">
                        Model: <b>{selected_model_name}</b> | Macro F1: <b>{active_cfg['macro_f1']*100:.2f}%</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander("🔍 Cleaned NLP Input"):
                    st.code(cleaned, language="text")
                    
            with col_chart:
                prob_df = pd.DataFrame({
                    'Class': [CLASS_INFO.get(c, {}).get('label', c) for c in CLASS_NAMES],
                    'Probability': probs * 100,
                    'Color': [CLASS_INFO.get(c, {}).get('color', '#334155') for c in CLASS_NAMES]
                }).sort_values(by='Probability', ascending=True)
                
                fig, ax = plt.subplots(figsize=(7, 4.2), dpi=200)
                bars = ax.barh(prob_df['Class'], prob_df['Probability'], color=prob_df['Color'], alpha=0.9, edgecolor='none')
                ax.set_xlim(0, max(prob_df['Probability'].max() * 1.25, 10))
                ax.set_xlabel("Confidence Probability (%)", fontsize=10, fontweight='bold')
                ax.tick_params(axis='both', labelsize=8.5)
                ax.grid(axis='x', linestyle='--', alpha=0.3)
                
                for bar in bars:
                    w = bar.get_width()
                    if w > 1.5:
                        ax.text(w + 0.8, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va='center', fontsize=8, fontweight='bold', color='#1e293b')
                        
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
            # Compare Across All Loaded Models
            with st.expander("⚡ Compare Prediction Across ALL Available Models"):
                comp_rows = []
                for m_name in AVAILABLE_MODELS:
                    b_bundle = load_model_bundle(m_name)
                    if b_bundle:
                        p_cls, p_cf, _, _ = predict_single(b_bundle, user_input)
                        comp_rows.append({
                            "Model": m_name,
                            "Family": AVAILABLE_MODELS[m_name]["category"],
                            "Benchmark Macro F1": f"{AVAILABLE_MODELS[m_name]['macro_f1']*100:.2f}%",
                            "Predicted Class": CLASS_INFO.get(p_cls, {}).get('label', p_cls),
                            "Confidence": f"{p_cf*100:.2f}%"
                        })
                if comp_rows:
                    st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

# ==========================================================================
# TAB 2: BATCH TESTING
# ==========================================================================
with tab_batch:
    st.markdown("### ⚡ Batch Testing & Test Split Evaluation")
    test_parquet_path = DATASET_DIR / "test_clean.parquet"
    
    source = st.radio("Choose Batch Source:", ["Sample from Test Dataset (15,160 Tweets)", "Upload CSV File"], horizontal=True)
    
    if source == "Sample from Test Dataset (15,160 Tweets)" and test_parquet_path.exists():
        df_test_full = pd.read_parquet(test_parquet_path)
        col_s1, col_s2 = st.columns([1, 2])
        with col_s1:
            n_samples = st.slider("Number of Samples:", 5, 50, 15, 5)
        with col_s2:
            class_filter = st.selectbox("Filter by True Class:", ["All Classes"] + [CLASS_INFO[c]["label"] for c in CLASS_NAMES])
            
        if st.button("🎲 Sample Random Test Tweets", type="primary"):
            label_col = "class_label" if "class_label" in df_test_full.columns else ("label" if "label" in df_test_full.columns else None)
            if class_filter != "All Classes" and label_col:
                filter_slug = [k for k, v in CLASS_INFO.items() if v["label"] == class_filter][0]
                sub = df_test_full[df_test_full[label_col] == filter_slug]
            else:
                sub = df_test_full
                
            sample_df = sub.sample(min(n_samples, len(sub)), random_state=np.random.randint(1, 10000)).copy()
            st.session_state["batch_data"] = sample_df
            
    elif source == "Upload CSV File":
        up_file = st.file_uploader("Upload CSV (must contain 'tweet_text' or 'clean_text' column):", type=['csv'])
        if up_file:
            st.session_state["batch_data"] = pd.read_csv(up_file)
            
    if "batch_data" in st.session_state and model_bundle:
        df_b = st.session_state["batch_data"].copy()
        text_col = "clean_text" if "clean_text" in df_b.columns else ("tweet_text" if "tweet_text" in df_b.columns else None)
        
        if text_col:
            st.markdown(f"#### Batch Predictions with **{selected_model_name}**")
            preds_list, confs_list = [], []
            for t in df_b[text_col]:
                p_c, c_f, _, _ = predict_single(model_bundle, str(t))
                preds_list.append(CLASS_INFO.get(p_c, {}).get("label", p_c))
                confs_list.append(f"{c_f*100:.1f}%")
                
            df_b["Predicted Class"] = preds_list
            df_b["Confidence"] = confs_list
            
            label_col = "class_label" if "class_label" in df_b.columns else ("label" if "label" in df_b.columns else None)
            cols_to_display = [text_col, "Predicted Class", "Confidence"]
            if label_col:
                df_b["Actual Class"] = [CLASS_INFO.get(c, {}).get("label", c) for c in df_b[label_col]]
                df_b["Correct?"] = np.where(df_b["Predicted Class"] == df_b["Actual Class"], "✅ Yes", "❌ No")
                acc = (df_b["Correct?"] == "✅ Yes").mean() * 100
                st.markdown(f"**Sample Accuracy**: **{acc:.1f}%** ({sum(df_b['Correct?'] == '✅ Yes')}/{len(df_b)} correct)")
                cols_to_display = [text_col, "Actual Class", "Predicted Class", "Confidence", "Correct?"]
                
            st.dataframe(df_b[cols_to_display], use_container_width=True, height=350)
            
            csv_data = df_b.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Predictions CSV", csv_data, "disaster_predictions.csv", "text/csv")

# ==========================================================================
# TAB 3: BENCHMARK LEADERBOARD & SUITE EXPLORER
# ==========================================================================
with tab_leaderboard:
    st.markdown("### 📊 Multi-Architecture Benchmark Suite Leaderboard")
    st.caption("Comparison across Classical ML, Word2Vec, FastText, GloVe, and Transformers on HumAID.")
    
    suite_tabs = st.tabs([
        "Classical ML (01)",
        "Word2Vec Recurrent (02)",
        "FastText Recurrent (03)",
        "GloVe Recurrent (04)",
        "BERT Transformer (05)",
        "RoBERTa Transformer (06)"
    ])
    
    suite_config = [
        ("01_classical_ml_tfidf_bow", "metrics_comparison.csv", "models_metrics_comparison.png", suite_tabs[0]),
        ("02_word2vec_recurrent_models", "summary/recurrent_models_benchmark_summary.csv", "summary/recurrent_models_benchmark_plot.png", suite_tabs[1]),
        ("03_fasttext_recurrent_models", "summary/recurrent_models_benchmark_summary.csv", "summary/recurrent_models_benchmark_plot.png", suite_tabs[2]),
        ("04_glove_recurrent_models", "summary/recurrent_models_benchmark_summary.csv", "summary/recurrent_models_benchmark_plot.png", suite_tabs[3]),
        ("05_transformer_bert", "models/bert_base/metrics.csv", "models/bert_base/training_curves.png", suite_tabs[4]),
        ("06_transformer_roberta", "models/roberta_base/metrics.csv", "models/roberta_base/training_curves.png", suite_tabs[5])
    ]
    
    for s_dir, csv_rel, plot_rel, s_tab in suite_config:
        with s_tab:
            s_csv = RESULTS_DIR / s_dir / csv_rel
            s_plot = RESULTS_DIR / s_dir / plot_rel
            
            if s_csv.exists() or s_plot.exists():
                c_p, c_t = st.columns([1.1, 0.9])
                with c_p:
                    if s_plot.exists():
                        st.image(str(s_plot), use_container_width=True)
                with c_t:
                    if s_csv.exists():
                        st.markdown("**Suite Performance Metrics:**")
                        df_s = pd.read_csv(s_csv)
                        st.dataframe(df_s, use_container_width=True)
                        
                # Show confusion matrices
                models_dir = RESULTS_DIR / s_dir / "models"
                if models_dir.exists():
                    st.markdown("#### 🎯 Per-Model Confusion Matrices & Per-Class Plots:")
                    m_dirs = [d for d in models_dir.iterdir() if d.is_dir()]
                    if m_dirs:
                        m_tabs = st.tabs([d.name.replace("_", " ").title() for d in m_dirs])
                        for d_entry, t_entry in zip(m_dirs, m_tabs):
                            with t_entry:
                                cm_p = d_entry / "confusion_matrix.png"
                                pc_p = d_entry / "per_class_metrics.png"
                                col1, col2 = st.columns(2)
                                with col1:
                                    if cm_p.exists():
                                        st.image(str(cm_p), use_container_width=True)
                                with col2:
                                    if pc_p.exists():
                                        st.image(str(pc_p), use_container_width=True)
            else:
                st.info(f"⏳ Benchmark for `{s_dir}` is currently executing on Kaggle GPU.")

# ==========================================================================
# TAB 4: DATASET & PREPROCESSING
# ==========================================================================
with tab_dataset:
    st.markdown("### 📖 HumAID Dataset & Preprocessing Pipeline")
    
    col_d1, col_d2 = st.columns([1, 1])
    with col_d1:
        st.markdown("""
        #### 📦 Dataset Architecture:
        - **Total Dataset Size**: 76,484 annotated disaster tweets
        - **Split Ratios**:
          - **Train Set**: 53,531 tweets (70%)
          - **Validation Set**: 7,793 tweets (10%)
          - **Test Set**: 15,160 tweets (20%)
        - **Class Imbalance**: Severe (59.56:1 ratio between majority and minority classes)
        """)
        
    with col_d2:
        st.markdown("""
        #### 🧹 Cleaning & Preprocessing Pipeline:
        1. **HTML Entity Unescaping**: `&amp;` $\\rightarrow$ `&`, `&lt;` $\\rightarrow$ `<`, etc.
        2. **Emoji Translation**: Translated 15+ disaster emojis to domain words (e.g. 🚨 $\\rightarrow$ `emergency alert`, 🌊 $\\rightarrow$ `flood water`).
        3. **URL & Mention Stripping**: Cleans noise tokens without loss of semantic meaning.
        4. **Hashtag Segmentation**: CamelCase splitting (e.g., `#FloodRelief` $\\rightarrow$ `flood relief`).
        5. **Contraction & Slang Normalization**: Normalized crisis shorthands (`pls` $\\rightarrow$ `please`, `emerg` $\\rightarrow$ `emergency`).
        """)
        
    st.markdown("---")
    st.markdown("#### 🎯 10 Humanitarian Target Categories:")
    c_cols = st.columns(5)
    for idx, (k, v) in enumerate(CLASS_INFO.items()):
        with c_cols[idx % 5]:
            st.markdown(f"""
            <div style="background:#ffffff; border-left:4px solid {v['color']}; padding:10px 14px; border-radius:6px; margin-bottom:10px; border-top:1px solid #e2e8f0; border-right:1px solid #e2e8f0; border-bottom:1px solid #e2e8f0;">
                <div style="font-size:1.1rem;">{v['icon']}</div>
                <div style="font-size:0.85rem; font-weight:700; color:{v['color']}; margin-top:2px;">{v['label']}</div>
            </div>
            """, unsafe_allow_html=True)
