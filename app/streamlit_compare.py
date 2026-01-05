# app/streamlit_compare.py
# Polished UI: cards, CSS, progress bars, metrics, example buttons, advanced settings expander,
# SVM label mapping, PLUS:
# - Cleaner “hero” input section (helper text + char counter + placeholder)
# - Predict/Clear/Random buttons on one line (form)
# - Lighter disclaimer (short line + full text inside expander)
# - Model status “chips” (loaded + device)
# - Better spacing + centered max width
# - FIX: Clear/Random works (dynamic widget key trick)
# - FIX: works on old/new Streamlit (safe_rerun)
# - Top-K label becomes dynamic: "Top-{k} prediction(s)"
# - OPTION 1 APPLIED: Removed “Final decision” banner completely
#
# Run:
#   python -m streamlit run app\streamlit_compare.py

import os
import json
import random
import numpy as np
import joblib
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =========================
# 1) Paths (EDIT IF NEEDED)
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROBERTA_DIR = os.path.join(BASE_DIR, "models", "roberta_export")
BERT_DIR    = os.path.join(BASE_DIR, "models", "bert_export")
SVM_DIR     = os.path.join(BASE_DIR, "models", "svm_export")

SVM_MODEL_PATH = os.path.join(SVM_DIR, "svm_calibrated_model.joblib")
SVM_VECT_PATH  = os.path.join(SVM_DIR, "tfidf_vectorizer.joblib")

MAX_LEN = 160
MAX_CHARS = 1200  # UI limit only (not model limit)

# =========================
# 2) Helpers
# =========================
def safe_rerun():
    """Compatibility: Streamlit old uses experimental_rerun(), newer uses rerun()."""
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_roberta(device):
    tok = AutoTokenizer.from_pretrained(ROBERTA_DIR)
    mdl = AutoModelForSequenceClassification.from_pretrained(ROBERTA_DIR).to(device)
    mdl.eval()
    with open(os.path.join(ROBERTA_DIR, "id2label.json"), "r", encoding="utf-8") as f:
        id2label = json.load(f)  # keys are strings "0","1",...
    return tok, mdl, id2label

@st.cache_resource
def load_bert(device):
    tok = AutoTokenizer.from_pretrained(BERT_DIR)
    mdl = AutoModelForSequenceClassification.from_pretrained(BERT_DIR).to(device)
    mdl.eval()
    id2label = {int(k): v for k, v in mdl.config.id2label.items()}
    return tok, mdl, id2label

@st.cache_resource
def load_svm():
    svm_model = joblib.load(SVM_MODEL_PATH)
    tfidf_vec = joblib.load(SVM_VECT_PATH)

    if not hasattr(svm_model, "predict_proba"):
        raise AttributeError("SVM model has no predict_proba(). Use a calibrated SVM.")
    if not hasattr(svm_model, "classes_"):
        raise AttributeError("SVM model has no classes_. Re-export a proper calibrated model.")

    classes = list(svm_model.classes_)
    return svm_model, tfidf_vec, classes

def predict_transformer(text, model, tokenizer, id2label, device, top_k=3, max_len=MAX_LEN, id2label_keys_are_str=False):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=max_len)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()

    top_idx = probs.argsort()[-top_k:][::-1]
    out = []
    for i in top_idx:
        key = str(int(i)) if id2label_keys_are_str else int(i)
        out.append((id2label[key], float(probs[int(i)])))
    return out

def predict_transformer_safe(
    text, model, tokenizer, id2label, device, top_k=3,
    min_proba=0.55, min_margin=0.15, max_len=MAX_LEN, id2label_keys_are_str=False
):
    topk = predict_transformer(
        text, model, tokenizer, id2label, device,
        top_k=top_k, max_len=max_len, id2label_keys_are_str=id2label_keys_are_str
    )
    top1 = topk[0][1]
    top2 = topk[1][1] if len(topk) > 1 else 0.0
    margin = top1 - top2

    if (top1 < min_proba) or (margin < min_margin):
        return {"label": "uncertain", "confidence": float(top1), "top_k": topk, "margin": float(margin)}
    return {"label": topk[0][0], "confidence": float(top1), "top_k": topk, "margin": float(margin)}

def svm_predict_proba(text, svm_model, tfidf_vec, svm_classes, top_k=3):
    Xv = tfidf_vec.transform([text])
    probs = svm_model.predict_proba(Xv)[0]
    top_idx = probs.argsort()[-top_k:][::-1]
    return [(svm_classes[int(i)], float(probs[int(i)])) for i in top_idx]

def svm_predict_safe_proba(text, svm_model, tfidf_vec, svm_classes, top_k=3, min_proba=0.55, min_margin=0.15):
    topk = svm_predict_proba(text, svm_model, tfidf_vec, svm_classes, top_k=top_k)
    top1 = topk[0][1]
    top2 = topk[1][1] if len(topk) > 1 else 0.0
    margin = top1 - top2

    if (top1 < min_proba) or (margin < min_margin):
        return {"label": "uncertain", "confidence": float(top1), "top_k": topk, "margin": float(margin)}
    return {"label": topk[0][0], "confidence": float(top1), "top_k": topk, "margin": float(margin)}

# =========================
# 3) UI helpers (CSS, cards, bars)
# =========================
def inject_css(theme="light"):
    if theme == "dark":
        bg = "#0B0F1A"
        card = "#111827"
        card2 = "#0F172A"
        text = "#E5E7EB"
        muted = "#94A3B8"
        border = "rgba(255,255,255,0.10)"
        accent = "#7C3AED"
        ok = "#22C55E"
        warn = "#F59E0B"
    else:
        bg = "#FFFFFF"
        card = "#FFFFFF"
        card2 = "#F8FAFC"
        text = "#0F172A"
        muted = "#475569"
        border = "rgba(2,6,23,0.10)"
        accent = "#4F46E5"
        ok = "#16A34A"
        warn = "#D97706"

    st.markdown(
        f"""
        <style>
        :root {{
          --bg: {bg};
          --card: {card};
          --card2: {card2};
          --text: {text};
          --muted: {muted};
          --border: {border};
          --accent: {accent};
          --ok: {ok};
          --warn: {warn};
        }}

        .stApp {{
          background: var(--bg);
          color: var(--text);
        }}

        .block-container {{
          max-width: 1100px;
          padding-top: 1.6rem;
          padding-bottom: 2.0rem;
        }}

        .app-title {{
          font-size: 2.0rem;
          font-weight: 800;
          margin: 0.2rem 0 0.2rem 0;
        }}
        .app-subtitle {{
          color: var(--muted);
          margin-bottom: 0.8rem;
        }}

        .chips {{
          display: flex;
          flex-wrap: wrap;
          gap: 8px;
          margin: 10px 0 12px 0;
        }}
        .chip {{
          padding: 6px 10px;
          border-radius: 999px;
          font-size: 0.82rem;
          border: 1px solid var(--border);
          background: var(--card2);
          color: var(--muted);
        }}
        .chip.ok {{
          color: var(--ok);
          border-color: rgba(34,197,94,0.25);
          background: rgba(34,197,94,0.08);
        }}

        .card {{
          background: var(--card);
          border: 1px solid var(--border);
          border-radius: 18px;
          padding: 16px 16px 12px 16px;
          box-shadow: 0 10px 25px rgba(0,0,0,0.06);
          margin-bottom: 12px;
        }}
        .card-header {{
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 10px;
        }}
        .card-title {{
          font-weight: 800;
          font-size: 1.1rem;
        }}
        .badge {{
          padding: 4px 10px;
          border-radius: 999px;
          font-size: 0.80rem;
          border: 1px solid var(--border);
          background: var(--card2);
          color: var(--muted);
        }}
        .badge.ok {{
          color: var(--ok);
          border-color: rgba(34,197,94,0.25);
          background: rgba(34,197,94,0.08);
        }}
        .badge.warn {{
          color: var(--warn);
          border-color: rgba(245,158,11,0.25);
          background: rgba(245,158,11,0.08);
        }}

        .muted {{ color: var(--muted); }}

        .row {{
          display: flex;
          gap: 10px;
          align-items: center;
          margin: 8px 0;
        }}
        .label {{
          width: 45%;
          font-size: 0.92rem;
        }}
        .pct {{
          width: 12%;
          text-align: right;
          color: var(--muted);
          font-size: 0.90rem;
        }}
        .bar {{
          width: 43%;
          height: 10px;
          border-radius: 999px;
          background: rgba(148,163,184,0.25);
          overflow: hidden;
          border: 1px solid var(--border);
        }}
        .bar > div {{
          height: 100%;
          background: var(--accent);
          border-radius: 999px;
        }}

        .note {{
          border: 1px dashed var(--border);
          border-radius: 14px;
          padding: 10px 12px;
          color: var(--muted);
          font-size: 0.92rem;
          margin: 10px 0 12px 0;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def status_badge(label: str):
    if label == "uncertain":
        return '<span class="badge warn">Uncertain</span>'
    return '<span class="badge ok">Confident</span>'

def render_topk_bars(topk_pairs):
    for lab, p in topk_pairs:
        pct = f"{p*100:.1f}%"
        width = max(0.0, min(100.0, p * 100.0))
        st.markdown(
            f"""
            <div class="row">
              <div class="label">{lab}</div>
              <div class="bar"><div style="width:{width:.2f}%"></div></div>
              <div class="pct">{pct}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

def topk_title(k: int) -> str:
    return f"Top-{k} prediction" if k == 1 else f"Top-{k} predictions"

def map_svm_label(raw_class, roberta_id2label):
    try:
        idx = int(raw_class)
        key = str(idx)
        if key in roberta_id2label:
            return roberta_id2label[key]
        return str(raw_class)
    except Exception:
        return str(raw_class)

# =========================
# 4) App
# =========================
st.set_page_config(page_title="Mental Health NLP — Comparator", layout="wide")

# Session defaults (IMPORTANT for Clear/Random)
if "ui_theme" not in st.session_state:
    st.session_state.ui_theme = "light"
if "draft_text" not in st.session_state:
    st.session_state["draft_text"] = ""
if "text_area_id" not in st.session_state:
    st.session_state["text_area_id"] = 0

EXAMPLES = [
    "I feel anxious and overwhelmed with everything lately.",
    "I keep checking the door again and again even when I know it's locked.",
    "I have lost interest in everything and feel empty every day.",
    "I'm doing okay recently, sleeping well and focusing on my routine.",
]

with st.sidebar:
    st.header("⚙️ Settings")
    st.session_state.ui_theme = st.radio("Theme", ["light", "dark"], index=0 if st.session_state.ui_theme == "light" else 1)
    top_k = st.slider("Top-K", 1, 5, 3)

    with st.expander("Advanced (SAFE thresholds)", expanded=False):
        min_proba = st.slider("Min probability", 0.10, 0.90, 0.55, 0.01)
        min_margin = st.slider("Min margin (top1-top2)", 0.00, 0.50, 0.15, 0.01)

    with st.expander("Quick examples", expanded=False):
        if st.button("Use example 1"):
            st.session_state["draft_text"] = EXAMPLES[0]
            st.session_state["text_area_id"] += 1
            safe_rerun()
        if st.button("Use example 2"):
            st.session_state["draft_text"] = EXAMPLES[1]
            st.session_state["text_area_id"] += 1
            safe_rerun()
        if st.button("Use example 3"):
            st.session_state["draft_text"] = EXAMPLES[2]
            st.session_state["text_area_id"] += 1
            safe_rerun()
        if st.button("Use example 4"):
            st.session_state["draft_text"] = EXAMPLES[3]
            st.session_state["text_area_id"] += 1
            safe_rerun()

    # Removed the "Show final decision" checkbox (since final decision is removed)
    show_debug_tables = st.checkbox("Show debug table", value=False)

inject_css(st.session_state.ui_theme)

st.markdown('<div class="app-title">🧠 Mental Health Detection — RoBERTa vs BERT vs Calibrated SVM</div>', unsafe_allow_html=True)
device = get_device()
st.markdown(f'<div class="app-subtitle">Device: <b>{device}</b> (CPU is OK for inference)</div>', unsafe_allow_html=True)

st.markdown('<div class="note"><b>Disclaimer:</b> Research/demo classifier — not a medical diagnosis tool.</div>', unsafe_allow_html=True)
with st.expander("Read full disclaimer"):
    st.write(
        "This app is a research/demo classifier and **not** a medical diagnosis tool. "
        "If you are worried about your mental wellbeing, consider talking to a trusted adult "
        "or a qualified professional."
    )

with st.spinner("Loading models..."):
    roberta_tok, roberta_mdl, roberta_id2label = load_roberta(device)
    bert_tok, bert_mdl, bert_id2label = load_bert(device)
    svm_mdl, tfidf_vec, svm_classes = load_svm()

st.markdown(
    f"""
    <div class="chips">
      <span class="chip ok">RoBERTa: loaded</span>
      <span class="chip ok">BERT: loaded</span>
      <span class="chip ok">SVM: loaded</span>
      <span class="chip">Device: {device}</span>
    </div>
    """,
    unsafe_allow_html=True
)

st.caption("Paste a sentence or a short social-media post. We’ll show Top-K probabilities for each model.")

# Dynamic key trick
text_key = f"draft_text_{st.session_state.text_area_id}"

with st.form("predict_form", clear_on_submit=False):
    text = st.text_area(
        "Enter a text:",
        height=140,
        value=st.session_state["draft_text"],
        key=text_key,
        placeholder="Example: I feel anxious and overwhelmed with everything lately..."
    )

    st.caption(f"{len(text)}/{MAX_CHARS} characters")
    if len(text) > MAX_CHARS:
        st.warning(f"Please keep the text under {MAX_CHARS} characters for a smoother UI experience.")

    b1, b2, b3 = st.columns([1, 1, 1])
    with b1:
        do_predict = st.form_submit_button("🔎 Predict", type="primary")
    with b2:
        do_clear = st.form_submit_button("🧹 Clear")
    with b3:
        do_random = st.form_submit_button("🎲 Random example")

# Sync widget value -> draft_text
st.session_state["draft_text"] = st.session_state.get(text_key, "")

if do_clear:
    st.session_state["draft_text"] = ""
    st.session_state["text_area_id"] += 1
    safe_rerun()

if do_random:
    st.session_state["draft_text"] = random.choice(EXAMPLES)
    st.session_state["text_area_id"] += 1
    safe_rerun()

st.markdown(
    "<div class='note'><b>SAFE policy:</b> We return <b>uncertain</b> when confidence is low "
    "(probability &lt; min_proba) or when the model hesitates (top1-top2 &lt; min_margin).</div>",
    unsafe_allow_html=True
)

if do_predict:
    text = st.session_state["draft_text"]
    if not text.strip():
        st.warning("Please enter a non-empty text.")
        st.stop()

    roberta_topk = predict_transformer(text, roberta_mdl, roberta_tok, roberta_id2label, device, top_k=top_k, id2label_keys_are_str=True)
    roberta_safe = predict_transformer_safe(text, roberta_mdl, roberta_tok, roberta_id2label, device, top_k=top_k, min_proba=min_proba, min_margin=min_margin, id2label_keys_are_str=True)

    bert_topk = predict_transformer(text, bert_mdl, bert_tok, bert_id2label, device, top_k=top_k, id2label_keys_are_str=False)
    bert_safe = predict_transformer_safe(text, bert_mdl, bert_tok, bert_id2label, device, top_k=top_k, min_proba=min_proba, min_margin=min_margin, id2label_keys_are_str=False)

    svm_topk_raw = svm_predict_proba(text, svm_mdl, tfidf_vec, svm_classes, top_k=top_k)
    svm_topk = [(map_svm_label(lab, roberta_id2label), p) for lab, p in svm_topk_raw]

    svm_safe_raw = svm_predict_safe_proba(text, svm_mdl, tfidf_vec, svm_classes, top_k=top_k, min_proba=min_proba, min_margin=min_margin)
    svm_safe = dict(svm_safe_raw)
    svm_safe["label"] = map_svm_label(svm_safe_raw["label"], roberta_id2label)
    svm_safe["top_k"] = [(map_svm_label(lab, roberta_id2label), p) for lab, p in svm_safe_raw["top_k"]]

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            f"""
            <div class="card">
              <div class="card-header">
                <div class="card-title">RoBERTa</div>
                {status_badge(roberta_safe["label"])}
              </div>
            """,
            unsafe_allow_html=True
        )
        st.metric("SAFE label", roberta_safe["label"])
        st.metric("Confidence", f"{roberta_safe['confidence']:.4f}")
        st.caption(f"Margin (top1-top2): {roberta_safe['margin']:.4f}")
        st.markdown(f"<div class='muted'><b>{topk_title(top_k)}</b></div>", unsafe_allow_html=True)
        render_topk_bars(roberta_topk)
        if show_debug_tables:
            st.dataframe([{"label": lab, "probability": float(p)} for lab, p in roberta_topk], use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown(
            f"""
            <div class="card">
              <div class="card-header">
                <div class="card-title">BERT</div>
                {status_badge(bert_safe["label"])}
              </div>
            """,
            unsafe_allow_html=True
        )
        st.metric("SAFE label", bert_safe["label"])
        st.metric("Confidence", f"{bert_safe['confidence']:.4f}")
        st.caption(f"Margin (top1-top2): {bert_safe['margin']:.4f}")
        st.markdown(f"<div class='muted'><b>{topk_title(top_k)}</b></div>", unsafe_allow_html=True)
        render_topk_bars(bert_topk)
        if show_debug_tables:
            st.dataframe([{"label": lab, "probability": float(p)} for lab, p in bert_topk], use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c3:
        st.markdown(
            f"""
            <div class="card">
              <div class="card-header">
                <div class="card-title">SVM (TF-IDF) — Calibrated</div>
                {status_badge(svm_safe["label"])}
              </div>
            """,
            unsafe_allow_html=True
        )
        st.metric("SAFE label", svm_safe["label"])
        st.metric("Confidence", f"{svm_safe['confidence']:.4f}")
        st.caption(f"Margin (top1-top2): {svm_safe['margin']:.4f}")
        st.markdown(f"<div class='muted'><b>{topk_title(top_k)}</b></div>", unsafe_allow_html=True)
        render_topk_bars(svm_topk)
        if show_debug_tables:
            st.dataframe([{"label": lab, "probability": float(p)} for lab, p in svm_topk], use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
