"""
Streamlit Web Application: Disaster Tweet Classification Benchmark
HumAID Dataset (76,484 Tweets across 10 Humanitarian Target Classes)
Features Top 5 Models for Inference + Deep Dive on Champion (BERT) + Comprehensive 24-Model Leaderboard
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
# Paths and Constant Metadata
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
# Preprocessing Engine (Exact Mirror of Notebook 00)
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
# Top 5 Model Loaders (Cached)
# --------------------------------------------------------------------------
TOP_5_CONFIG = {
    "BERT Base Uncased": {
        "rank": 1,
        "type": "transformer",
        "path": RESULTS_DIR / "05_transformer_bert" / "saved_models",
        "macro_f1": 0.7540,
        "accuracy": 0.7710,
        "category": "Transformer (BERT)",
        "params": "110M"
    },
    "RoBERTa Base": {
        "rank": 2,
        "type": "transformer",
        "path": RESULTS_DIR / "06_transformer_roberta" / "saved_models",
        "macro_f1": 0.7529,
        "accuracy": 0.7710,
        "category": "Transformer (RoBERTa)",
        "params": "125M"
    },
    "Word TF-IDF + Logistic Regression": {
        "rank": 3,
        "type": "classical_sklearn",
        "vec_path": RESULTS_DIR / "01_classical_ml_tfidf_bow" / "saved_models" / "word_tfidf_vectorizer.joblib",
        "model_path": RESULTS_DIR / "01_classical_ml_tfidf_bow" / "saved_models" / "word_tf_idf_plus_logisticregression.joblib",
        "macro_f1": 0.7172,
        "accuracy": 0.7363,
        "category": "Classical ML",
        "params": "20K Features"
    },
    "Char TF-IDF + LinearSVC": {
        "rank": 4,
        "type": "classical_sklearn_decision",
        "vec_path": RESULTS_DIR / "01_classical_ml_tfidf_bow" / "saved_models" / "char_tfidf_vectorizer.joblib",
        "model_path": RESULTS_DIR / "01_classical_ml_tfidf_bow" / "saved_models" / "char_tf_idf_plus_linearsvc.joblib",
        "macro_f1": 0.7120,
        "accuracy": 0.7362,
        "category": "Classical ML",
        "params": "30K Char-ngrams"
    },
    "Char TF-IDF + Logistic Regression": {
        "rank": 5,
        "type": "classical_sklearn",
        "vec_path": RESULTS_DIR / "01_classical_ml_tfidf_bow" / "saved_models" / "char_tfidf_vectorizer.joblib",
        "model_path": RESULTS_DIR / "01_classical_ml_tfidf_bow" / "saved_models" / "char_tf_idf_plus_logisticregression.joblib",
        "macro_f1": 0.7112,
        "accuracy": 0.7336,
        "category": "Classical ML",
        "params": "30K Char-ngrams"
    }
}

@st.cache_resource
def load_top_model(model_name: str):
    cfg = TOP_5_CONFIG[model_name]
    if cfg["type"] == "transformer":
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        tok = AutoTokenizer.from_pretrained(cfg["path"])
        model = AutoModelForSequenceClassification.from_pretrained(cfg["path"])
        model.eval()
        return {"type": "transformer", "tok": tok, "model": model}
    elif cfg["type"] in ["classical_sklearn", "classical_sklearn_decision"]:
        vec = joblib.load(cfg["vec_path"])
        clf = joblib.load(cfg["model_path"])
        return {"type": cfg["type"], "vec": vec, "clf": clf}

def predict_single(model_bundle, raw_text: str):
    cleaned = clean_tweet_text(raw_text)
    if not cleaned:
        return None, 0.0, np.zeros(len(CLASS_NAMES)), ""
        
    m_type = model_bundle["type"]
    if m_type == "transformer":
        tok = model_bundle["tok"]
        model = model_bundle["model"]
        inputs = tok([cleaned], padding=True, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    elif m_type == "classical_sklearn":
        vec = model_bundle["vec"]
        clf = model_bundle["clf"]
        x = vec.transform([cleaned])
        probs = clf.predict_proba(x)[0]
    elif m_type == "classical_sklearn_decision":
        vec = model_bundle["vec"]
        clf = model_bundle["clf"]
        x = vec.transform([cleaned])
        df_scores = clf.decision_function(x)[0]
        # Softmax for probability approximation
        exp_s = np.exp(df_scores - np.max(df_scores))
        probs = exp_s / exp_s.sum()
        
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
    selected_model_name = st.selectbox(
        "Choose Top 5 Model for Inference:",
        list(TOP_5_CONFIG.keys()),
        index=0,
        help="Select any of the top 5 models benchmarked across the HumAID dataset."
    )
    
    active_cfg = TOP_5_CONFIG[selected_model_name]
    
    st.markdown(f"""
    <div style="background:#f8fafc; padding:12px; border-radius:8px; border:1px solid #e2e8f0; margin-top:8px;">
        <div style="font-size:0.8rem; color:#64748b; font-weight:600;">LEADERBOARD RANK: <span style="color:#0f172a;">#{active_cfg['rank']}</span></div>
        <div style="font-size:0.9rem; font-weight:700; color:#0f172a; margin-top:2px;">{selected_model_name}</div>
        <div style="font-size:0.8rem; color:#475569; margin-top:4px;">Architecture: <b>{active_cfg['category']}</b></div>
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
    
    st.markdown("---")
    st.markdown("#### 📚 Dataset Benchmark Context")
    st.markdown("- **HumAID Corpus**: 76,484 tweets")
    st.markdown("- **Test Split**: 15,160 tweets")
    st.markdown("- **Classes**: 10 Crisis Categories")
    st.markdown("- **Total Models Tested**: 24 Models")
    
    st.markdown("---")
    st.caption("CSE 4122 NLP Lab Project | Md. Nayeem & MD Jahid Hasan Jim")

# Load active model bundle
model_bundle = load_top_model(selected_model_name)

# --------------------------------------------------------------------------
# Main App Header
# --------------------------------------------------------------------------
st.markdown(f"""
<div class="main-header">
    <div class="main-title">
        <span>🚨 Disaster Tweet Classification System</span>
    </div>
    <div class="main-subtitle">
        End-to-end multi-model NLP benchmark & real-time crisis response classifier evaluated on 76,484 HumAID tweets.
    </div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Tabs Architecture
# --------------------------------------------------------------------------
tab_classifier, tab_batch, tab_champion, tab_leaderboard, tab_dataset = st.tabs([
    "🔮 Live Classifier",
    "⚡ Batch Testing",
    "🌟 Best Model Deep-Dive (BERT)",
    "📊 All 24 Results & Leaderboard",
    "📖 Dataset & Preprocessing"
])

# ==========================================================================
# TAB 1: LIVE CLASSIFIER (TOP 5 MODELS)
# ==========================================================================
with tab_classifier:
    st.markdown("### 💬 Single Tweet Classification & Multi-Class Confidence")
    st.caption(f"Active Inference Model: **{selected_model_name}** (Rank #{active_cfg['rank']} Leaderboard Champion)")
    
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
            
    if (classify_btn or user_input.strip()) and user_input.strip():
        with st.spinner("Processing & predicting..."):
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
                
                with st.expander("🔍 Cleaned NLP Representation"):
                    st.markdown("**Cleaned Input (Noise & Slang Removed):**")
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
                
            # Optional Multi-Model Comparison on the Same Tweet
            with st.expander("⚡ Compare Prediction Across ALL Top 5 Models"):
                comp_rows = []
                for m_name in TOP_5_CONFIG:
                    b_bundle = load_top_model(m_name)
                    p_cls, p_cf, _, _ = predict_single(b_bundle, user_input)
                    comp_rows.append({
                        "Rank": f"#{TOP_5_CONFIG[m_name]['rank']}",
                        "Model": m_name,
                        "Category": TOP_5_CONFIG[m_name]["category"],
                        "Macro F1": f"{TOP_5_CONFIG[m_name]['macro_f1']*100:.2f}%",
                        "Predicted Class": CLASS_INFO.get(p_cls, {}).get('label', p_cls),
                        "Confidence": f"{p_cf*100:.2f}%"
                    })
                st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

# ==========================================
# TAB 2: BATCH TESTING
# ==========================================
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
            if class_filter != "All Classes":
                filter_slug = [k for k, v in CLASS_INFO.items() if v["label"] == class_filter][0]
                sub = df_test_full[df_test_full["class_label"] == filter_slug]
            else:
                sub = df_test_full
                
            sample_df = sub.sample(min(n_samples, len(sub)), random_state=np.random.randint(1, 10000)).copy()
            st.session_state["batch_data"] = sample_df
            
    elif source == "Upload CSV File":
        up_file = st.file_uploader("Upload CSV (must contain 'tweet_text' or 'clean_text' column):", type=['csv'])
        if up_file:
            st.session_state["batch_data"] = pd.read_csv(up_file)
            
    if "batch_data" in st.session_state:
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
            
            cols_to_display = [text_col, "Predicted Class", "Confidence"]
            if "class_label" in df_b.columns:
                df_b["Actual Class"] = [CLASS_INFO.get(c, {}).get("label", c) for c in df_b["class_label"]]
                df_b["Correct?"] = np.where(df_b["Predicted Class"] == df_b["Actual Class"], "✅ Yes", "❌ No")
                acc = (df_b["Correct?"] == "✅ Yes").mean() * 100
                st.markdown(f"**Sample Accuracy**: **{acc:.1f}%** ({sum(df_b['Correct?'] == '✅ Yes')}/{len(df_b)} correct)")
                cols_to_display = [text_col, "Actual Class", "Predicted Class", "Confidence", "Correct?"]
                
            st.dataframe(df_b[cols_to_display], use_container_width=True, height=350)
            
            csv_data = df_b.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Predictions CSV", csv_data, "disaster_predictions.csv", "text/csv")

# ==========================================================================
# TAB 3: DEEP-DIVE ON BEST MODEL (BERT BASE UNCASED)
# ==========================================================================
with tab_champion:
    st.markdown("### 🌟 Deep Dive: Champion Model — BERT Base Uncased")
    st.caption("Detailed architectural analysis, performance breakdown, and per-class diagnostic evaluation.")
    
    # 1. Metric cards
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown("""<div class="metric-card"><div style="font-size:0.8rem; color:#64748b; font-weight:600;">TEST MACRO F1</div><div style="font-size:1.6rem; font-weight:800; color:#1f77b4;">75.40%</div><div style="font-size:0.75rem; color:#16a34a; font-weight:600;">Rank #1 in Benchmark</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="metric-card"><div style="font-size:0.8rem; color:#64748b; font-weight:600;">TEST ACCURACY</div><div style="font-size:1.6rem; font-weight:800; color:#ff7f0e;">77.10%</div><div style="font-size:0.75rem; color:#64748b;">11,688 / 15,160 Test</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="metric-card"><div style="font-size:0.8rem; color:#64748b; font-weight:600;">WEIGHTED F1</div><div style="font-size:1.6rem; font-weight:800; color:#0f172a;">76.78%</div><div style="font-size:0.75rem; color:#64748b;">Frequency-Weighted</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown("""<div class="metric-card"><div style="font-size:0.8rem; color:#64748b; font-weight:600;">MACRO RECALL</div><div style="font-size:1.6rem; font-weight:800; color:#0f172a;">78.15%</div><div style="font-size:0.75rem; color:#64748b;">Crisis Sensitivity</div></div>""", unsafe_allow_html=True)
    with c5:
        st.markdown("""<div class="metric-card"><div style="font-size:0.8rem; color:#64748b; font-weight:600;">MACRO PRECISION</div><div style="font-size:1.6rem; font-weight:800; color:#0f172a;">73.53%</div><div style="font-size:0.75rem; color:#64748b;">Positive Predictive Val</div></div>""", unsafe_allow_html=True)
        
    st.markdown("---")
    
    # 2. Confusion Matrix & Per Class Metrics
    col_cm, col_pc = st.columns([1.1, 0.9])
    
    bert_cm_file = RESULTS_DIR / "05_transformer_bert" / "models" / "bert_base_uncased" / "confusion_matrix.png"
    bert_pc_file = RESULTS_DIR / "05_transformer_bert" / "models" / "bert_base_uncased" / "per_class_metrics.png"
    bert_tc_file = RESULTS_DIR / "05_transformer_bert" / "models" / "bert_base_uncased" / "training_curves.png"
    
    with col_cm:
        st.markdown("#### 🎯 Dual Confusion Matrix (Raw Counts & Normalized %)")
        if bert_cm_file.exists():
            st.image(str(bert_cm_file), use_container_width=True)
            
    with col_pc:
        st.markdown("#### 📊 Per-Class Performance Bar Chart")
        if bert_pc_file.exists():
            st.image(str(bert_pc_file), use_container_width=True)
            
    st.markdown("---")
    
    # 3. Training Progression & Per Class Table
    col_tc, col_rep = st.columns([1, 1])
    with col_tc:
        st.markdown("#### 📈 Fine-Tuning Convergence Curves")
        if bert_tc_file.exists():
            st.image(str(bert_tc_file), use_container_width=True)
            
    with col_rep:
        st.markdown("#### 📋 Detailed Per-Class Breakdown Table")
        bert_pc_csv = RESULTS_DIR / "05_transformer_bert" / "models" / "bert_base_uncased" / "per_class_metrics.csv"
        if bert_pc_csv.exists():
            df_pc = pd.read_csv(bert_pc_csv)
            df_pc["Class"] = [CLASS_INFO.get(c, {}).get("label", c) for c in df_pc["Class"]]
            df_pc["Precision"] = df_pc["Precision"].map(lambda x: f"{x*100:.2f}%")
            df_pc["Recall"] = df_pc["Recall"].map(lambda x: f"{x*100:.2f}%")
            df_pc["F1-Score"] = df_pc["F1-Score"].map(lambda x: f"{x*100:.2f}%")
            df_pc["Support"] = df_pc["Support"].astype(int)
            st.dataframe(df_pc, use_container_width=True, hide_index=True)
            
    with st.expander("📄 Full Text Classification Report (BERT Base)"):
        bert_rep_txt = RESULTS_DIR / "05_transformer_bert" / "models" / "bert_base_uncased" / "classification_report.txt"
        if bert_rep_txt.exists():
            st.code(bert_rep_txt.read_text(encoding='utf-8'), language="text")

# ==========================================================================
# TAB 4: ALL 24 RESULTS & MASTER LEADERBOARD
# ==========================================================================
with tab_leaderboard:
    st.markdown("### 📊 Comprehensive 24-Model Benchmark Leaderboard")
    st.caption("Benchmarked across 6 Model Families: Classical ML, Word2Vec, FastText, GloVe, BERT, and RoBERTa on HumAID.")
    
    master_csv_path = RESULTS_DIR / "v2_master_metrics_summary.csv"
    master_plot_path = RESULTS_DIR / "master_models_comparison.png"
    
    # 1. Master Grouped Comparison Figure
    if master_plot_path.exists():
        st.markdown("#### 🏆 Master Comparative Benchmark (Macro F1 vs Accuracy Across All 24 Models)")
        st.image(str(master_plot_path), use_container_width=True)
        
    st.markdown("---")
    
    # 2. Interactive Leaderboard Table
    if master_csv_path.exists():
        df_master = pd.read_csv(master_csv_path, index_col=0)
        
        col_f1, col_f2 = st.columns([1, 2])
        with col_f1:
            cat_filter = st.selectbox("Filter by Category:", ["All Categories"] + sorted(df_master["Category"].unique().tolist()))
            
        filtered_df = df_master if cat_filter == "All Categories" else df_master[df_master["Category"] == cat_filter]
        
        display_df = filtered_df.copy()
        for col in ["Test Macro F1", "Test Accuracy", "Test Weighted F1", "Test Macro Precision", "Test Macro Recall", "Val Best Macro F1"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].map(lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "-")
                
        st.dataframe(display_df, use_container_width=True)
        
    st.markdown("---")
    
    # 3. Suite-by-Suite Comparison Explorer
    st.markdown("#### 🔬 Suite-by-Suite Detailed Comparison Explorer")
    suite_tabs = st.tabs([
        "Classical ML (10 Models)",
        "Word2Vec Recurrent (6 Models)",
        "FastText Recurrent (3 Models)",
        "GloVe Recurrent (3 Models)",
        "BERT Transformer",
        "RoBERTa Transformer"
    ])
    
    suite_map = [
        ("01_classical_ml_tfidf_bow", suite_tabs[0]),
        ("02_word2vec_recurrent_models", suite_tabs[1]),
        ("03_fasttext_recurrent_models", suite_tabs[2]),
        ("04_glove_recurrent_models", suite_tabs[3]),
        ("05_transformer_bert", suite_tabs[4]),
        ("06_transformer_roberta", suite_tabs[5])
    ]
    
    for s_dir, s_tab in suite_map:
        with s_tab:
            s_plot = RESULTS_DIR / s_dir / "models_metrics_comparison.png"
            s_csv = RESULTS_DIR / s_dir / "metrics_comparison.csv"
            
            c_p, c_t = st.columns([1.2, 0.8])
            with c_p:
                if s_plot.exists():
                    st.image(str(s_plot), use_container_width=True)
            with c_t:
                if s_csv.exists():
                    st.markdown("**Suite Metrics Table:**")
                    st.dataframe(pd.read_csv(s_csv), use_container_width=True)

# ==========================================================================
# TAB 5: DATASET & PREPROCESSING
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
