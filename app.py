"""
Streamlit Web Application: Disaster Tweet Classification
Model: Sublinear TF-IDF + Balanced Logistic Regression
Dataset: HumAID (10 Humanitarian Target Classes)
Course: CSE 4122 (NLP Lab)
Authors: Md. Nayeem (2107050) & MD Jahid Hasan Jim (2107054)
"""

import html
import re
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------
# Page Configuration & Minimal Styling
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Disaster Tweet Classifier",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .header-box {
        padding: 18px 22px;
        background: #0f172a;
        color: #f8fafc;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    .header-title {
        font-size: 1.45rem;
        font-weight: 700;
        margin: 0;
        color: #f8fafc;
    }
    .header-sub {
        font-size: 0.9rem;
        color: #94a3b8;
        margin-top: 4px;
        margin-bottom: 0;
    }
    .result-box {
        border-radius: 8px;
        padding: 16px 20px;
        margin-top: 10px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
    }
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 600;
        color: #ffffff;
    }
    .confidence-score {
        font-size: 1.5rem;
        font-weight: 700;
        color: #0f172a;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Text Preprocessing
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

CLASS_INFO = {
    "caution_and_advice": {"label": "Caution & Advice", "color": "#d97706"},
    "displaced_people_and_evacuations": {"label": "Displaced People & Evacuations", "color": "#7c3aed"},
    "infrastructure_and_utility_damage": {"label": "Infrastructure & Utility Damage", "color": "#475569"},
    "injured_or_dead_people": {"label": "Injured or Dead People", "color": "#dc2626"},
    "missing_or_found_people": {"label": "Missing or Found People", "color": "#ea580c"},
    "not_humanitarian": {"label": "Not Humanitarian", "color": "#64748b"},
    "other_relevant_information": {"label": "Other Relevant Information", "color": "#0284c7"},
    "requests_or_urgent_needs": {"label": "Requests & Urgent Needs", "color": "#b91c1c"},
    "rescue_volunteering_or_donation_effort": {"label": "Rescue, Volunteering & Donations", "color": "#16a34a"},
    "sympathy_and_support": {"label": "Sympathy & Support", "color": "#db2777"}
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


@st.cache_resource
def load_model():
    model_path = Path("results/model_1/model.joblib")
    if not model_path.exists():
        st.error(f"Model checkpoint not found at `{model_path}`. Run `python model_1_tfidf_lr.py` first.")
        st.stop()
    return joblib.load(model_path)


checkpoint = load_model()
clf = checkpoint['classifier']
tfidf = checkpoint['vectorizer']
class_names = checkpoint['class_names']
test_metrics = checkpoint.get('test_metrics', {})

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.subheader("Disaster Tweet Classifier")
    st.caption("CSE 4122 NLP Lab Project")
    st.markdown("---")
    
    st.markdown("**Model**: TF-IDF + Logistic Regression")
    st.markdown("**Features**: 20,000 unigrams + bigrams")
    st.markdown("**Weighting**: Balanced Class Weights")
    st.markdown("---")
    
    st.markdown("**Test Set Performance**")
    st.metric("Macro F1", f"{test_metrics.get('macro_f1', 0.7138)*100:.2f}%")
    st.metric("Accuracy", f"{test_metrics.get('accuracy', 0.7348)*100:.2f}%")
    st.metric("Macro Recall", f"{test_metrics.get('macro_recall', 0.7427)*100:.2f}%")
    st.metric("Macro Precision", f"{test_metrics.get('macro_precision', 0.6943)*100:.2f}%")
    st.markdown("---")
    st.caption("Md. Nayeem (2107050) & MD Jahid Hasan Jim (2107054)")

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown("""
<div class="header-box">
    <div class="header-title">Disaster Tweet Classification System</div>
    <div class="header-sub">Humanitarian crisis text categorization using TF-IDF & Balanced Logistic Regression (HumAID Dataset)</div>
</div>
""", unsafe_allow_html=True)

tab_single, tab_batch, tab_perf, tab_about = st.tabs([
    "Classifier",
    "Batch Testing",
    "Model Performance",
    "Dataset & Methodology"
])

# ==========================================================================
# TAB 1: Single Classifier
# ==========================================================================
with tab_single:
    PRESETS = {
        "Custom Input": "",
        "Urgent Request": "SOS! We are trapped on the second floor on Pine St due to flood waters. Need clean drinking water and baby formula urgently #HurricaneHarvey",
        "Missing Person": "MISSING: 7-year-old Lucas Vance last seen wearing blue raincoat near Spring Creek after the flood. Contact emergency dispatch #MissingPerson",
        "Evacuation & Shelter": "Over 2,500 residents evacuated from Riverside community are now housed at the high school gymnasium shelter #FloodEvacuation",
        "Infrastructure Damage": "Power grid failed across 4 districts. Main bridge on Highway 10 is cracked and impassable due to landslides #EarthquakeDamage",
        "Caution & Advice": "FLASH FLOOD WARNING: Move to higher ground immediately. Do not drive through flooded roads. Follow local emergency broadcasts.",
        "Injuries & Casualties": "Officials report at least 14 injured and 3 dead following building collapse after 6.8 magnitude earthquake.",
        "Rescue & Volunteering": "Red Cross boats and volunteer crews delivered 500 meals and hygiene kits to cut-off neighborhoods today #DisasterRelief",
        "Sympathy & Support": "Sending prayers and heartfelt condolences to all families affected by the storm in Florida. Stay strong.",
        "Not Humanitarian": "Having pizza while watching the new movie on Netflix tonight with friends."
    }
    
    selected_preset = st.selectbox("Preset Example:", list(PRESETS.keys()))
    default_val = PRESETS[selected_preset]
    
    user_text = st.text_area(
        "Tweet Text:",
        value=default_val,
        height=90,
        placeholder="Enter or paste tweet text..."
    )
    
    col_btn, col_len = st.columns([1, 4])
    with col_btn:
        run_btn = st.button("Classify", type="primary")
    with col_len:
        if user_text:
            st.caption(f"{len(user_text)} characters | {len(user_text.split())} words")
            
    if (run_btn or user_text) and user_text.strip():
        cleaned = clean_tweet_text(user_text)
        vec = tfidf.transform([cleaned])
        probs = clf.predict_proba(vec)[0]
        pred_idx = int(np.argmax(probs))
        pred_raw = class_names[pred_idx]
        conf = probs[pred_idx]
        
        info = CLASS_INFO.get(pred_raw, {"label": pred_raw.replace('_', ' ').title(), "color": "#0f172a"})
        
        st.markdown("---")
        col_res, col_chart = st.columns([1, 1.4])
        
        with col_res:
            st.markdown(f"""
            <div class="result-box" style="border-left: 5px solid {info['color']};">
                <div style="font-size:0.85rem; color:#64748b; margin-bottom:4px;">Predicted Category</div>
                <div style="font-size:1.25rem; font-weight:700; color:{info['color']}; margin-bottom:12px;">
                    {info['label']}
                </div>
                <div style="font-size:0.85rem; color:#64748b; margin-bottom:2px;">Confidence</div>
                <div class="confidence-score">{conf * 100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("Preprocessing Details"):
                st.markdown("**Cleaned Input:**")
                st.code(cleaned, language="text")
        
        with col_chart:
            prob_df = pd.DataFrame({
                'Class': [CLASS_INFO.get(c, {}).get('label', c) for c in class_names],
                'Probability': probs * 100
            }).sort_values(by='Probability', ascending=True)
            
            fig, ax = plt.subplots(figsize=(6, 3.8))
            bars = ax.barh(prob_df['Class'], prob_df['Probability'], color="#334155", edgecolor="#0f172a", alpha=0.85)
            ax.set_xlim(0, max(prob_df['Probability'].max() * 1.2, 10))
            ax.set_xlabel("Probability (%)", fontsize=9)
            ax.tick_params(axis='both', labelsize=8.5)
            for bar in bars:
                w = bar.get_width()
                if w > 2:
                    ax.text(w + 1, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va='center', fontsize=8, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

# ==========================================================================
# TAB 2: Batch Testing
# ==========================================================================
with tab_batch:
    test_file = Path("dataset/test.parquet")
    
    source = st.radio("Source:", ["Test Dataset Sample", "Upload CSV"], horizontal=True)
    batch_df = None
    
    if source == "Test Dataset Sample" and test_file.exists():
        df_all = pd.read_parquet(test_file)
        col_s, col_f = st.columns([1, 2])
        with col_s:
            n_samples = st.slider("Samples:", 5, 50, 15, 5)
        with col_f:
            cls_filter = st.selectbox("Filter Class:", ["All"] + sorted(df_all['class_label'].unique()))
            
        if st.button("Sample & Predict"):
            sub = df_all if cls_filter == "All" else df_all[df_all['class_label'] == cls_filter]
            batch_df = sub.sample(min(n_samples, len(sub)), random_state=np.random.randint(1, 10000)).copy()
            st.session_state['eval_batch'] = batch_df
            
    elif source == "Upload CSV":
        up = st.file_uploader("Upload CSV (must contain 'tweet_text' or 'text' column):", type=['csv'])
        if up is not None:
            batch_df = pd.read_csv(up)
            st.session_state['eval_batch'] = batch_df
            
    if 'eval_batch' in st.session_state:
        df_b = st.session_state['eval_batch'].copy()
        t_col = 'tweet_text' if 'tweet_text' in df_b.columns else ('text' if 'text' in df_b.columns else None)
        
        if t_col:
            df_b['clean'] = df_b[t_col].apply(clean_tweet_text)
            X_b = tfidf.transform(df_b['clean'])
            b_probs = clf.predict_proba(X_b)
            b_idx = np.argmax(b_probs, axis=1)
            b_conf = np.max(b_probs, axis=1)
            
            df_b['Predicted'] = [CLASS_INFO.get(class_names[i], {}).get('label', class_names[i]) for i in b_idx]
            df_b['Confidence'] = [f"{c*100:.1f}%" for c in b_conf]
            
            cols_show = [t_col, 'Predicted', 'Confidence']
            if 'class_label' in df_b.columns:
                df_b['Actual'] = [CLASS_INFO.get(c, {}).get('label', c) for c in df_b['class_label']]
                df_b['Match'] = np.where(df_b['class_label'] == [class_names[i] for i in b_idx], "Yes", "No")
                acc = (df_b['Match'] == "Yes").mean() * 100
                st.markdown(f"**Batch Accuracy**: **{acc:.1f}%** ({sum(df_b['Match'] == 'Yes')}/{len(df_b)})")
                cols_show = [t_col, 'Actual', 'Predicted', 'Confidence', 'Match']
                
            st.dataframe(df_b[cols_show], height=350)
            
            csv_bytes = df_b.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv_bytes, "predictions.csv", "text/csv")

# ==========================================================================
# TAB 3: Model Performance
# ==========================================================================
with tab_perf:
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Macro F1", "71.38%")
    with col_m2:
        st.metric("Accuracy", "73.48%")
    with col_m3:
        st.metric("Macro Recall", "74.27%")
    with col_m4:
        st.metric("Macro Precision", "69.43%")
        
    st.markdown("---")
    
    col_p1, col_p2 = st.columns(2)
    cm_path = Path("results/model_1/confusion_matrix_normalized.png")
    f1_path = Path("results/model_1/per_class_f1_score.png")
    
    with col_p1:
        st.markdown("**Normalized Confusion Matrix**")
        if cm_path.exists():
            st.image(str(cm_path))
    with col_p2:
        st.markdown("**Per-Class F1-Score**")
        if f1_path.exists():
            st.image(str(f1_path))
            
    rep_file = Path("results/model_1/classification_report.txt")
    if rep_file.exists():
        with st.expander("Full Classification Report"):
            st.code(rep_file.read_text(encoding='utf-8'), language='text')

# ==========================================================================
# TAB 4: Dataset & Methodology
# ==========================================================================
with tab_about:
    st.markdown("""
    **Dataset**: HumAID (QCRI / HumAID-all) — 76,484 annotated disaster tweets across 10 classes.
    - Train: 53,531 (70%) | Val: 7,793 (10%) | Test: 15,160 (20%)
    - Class Imbalance: 59.56:1 ratio (Majority: 14,891 vs Minority: 250)
    
    **Optimization Strategy**:
    - Inverse class frequency weights $w_c = \\frac{N}{|C| \\cdot N_c}$ for balanced Macro F1.
    - Sublinear TF-IDF (1-2 ngrams, 20,000 features, $C=2.0$).
    
    **Preprocessing**:
    - HTML entity decoding, emoji translation to disaster keywords, CamelCase hashtag parsing, contraction & disaster slang normalization.
    """)
