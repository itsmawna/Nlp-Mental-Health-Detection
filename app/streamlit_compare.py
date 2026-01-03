# app/streamlit_compare.py
# Compare RoBERTa vs BERT vs Calibrated SVM (TF-IDF) — show CLASS NAMES (not ids)
# Run: streamlit run app/streamlit_compare.py

import os
import json
import numpy as np
import joblib
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =========================
# 1) Paths (EDIT IF NEEDED)
# =========================
ROBERTA_DIR = r"C:\Users\21276\NLP-Mental-Health-Detection\models\roberta_export"
BERT_DIR    = r"C:\Users\21276\NLP-Mental-Health-Detection\models\bert_export"
SVM_DIR     = r"C:\Users\21276\NLP-Mental-Health-Detection\models\svm_export"

SVM_MODEL_PATH = os.path.join(SVM_DIR, "svm_calibrated_model.joblib")
SVM_VECT_PATH  = os.path.join(SVM_DIR, "tfidf_vectorizer.joblib")

MAX_LEN = 160

# =========================
# 2) Helpers
# =========================
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

    classes = list(svm_model.classes_)  # can be strings OR ints
    return svm_model, tfidf_vec, classes

def topk_to_table(topk, value_name="value"):
    return [{"label": lab, value_name: float(val)} for lab, val in topk]

# -------------------------
# Transformers: predict + safe
# -------------------------
def predict_transformer(text, model, tokenizer, id2label, device,
                        top_k=3, max_len=MAX_LEN, id2label_keys_are_str=False):
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

def predict_transformer_safe(text, model, tokenizer, id2label, device,
                             top_k=3, min_proba=0.55, min_margin=0.15,
                             max_len=MAX_LEN, id2label_keys_are_str=False):
    topk = predict_transformer(
        text, model, tokenizer, id2label, device,
        top_k=top_k, max_len=max_len, id2label_keys_are_str=id2label_keys_are_str
    )
    top1 = topk[0][1]
    top2 = topk[1][1] if len(topk) > 1 else 0.0
    margin = top1 - top2

    if (top1 < min_proba) or (margin < min_margin):
        return {"label": "uncertain", "confidence": float(top1), "top_k": topk}
    return {"label": topk[0][0], "confidence": float(top1), "top_k": topk}

# -------------------------
# ✅ Calibrated SVM: predict + safe (probabilities) WITH CLASS NAMES
# - If svm_classes are strings -> use directly
# - If svm_classes are ints -> map using GLOBAL_ID2LABEL (from RoBERTa export)
# -------------------------
@st.cache_resource
def load_global_id2label():
    # We reuse RoBERTa mapping as the global mapping for SVM if needed
    with open(os.path.join(ROBERTA_DIR, "id2label.json"), "r", encoding="utf-8") as f:
        d = json.load(f)  # keys "0","1",...
    return d

GLOBAL_ID2LABEL = load_global_id2label()

def _svm_class_to_name(cls):
    # cls can be 'depression' (str) OR 6 (int)
    if isinstance(cls, (int, np.integer)):
        return GLOBAL_ID2LABEL.get(str(int(cls)), str(int(cls)))
    return str(cls)

def svm_predict_proba(text, svm_model, tfidf_vec, svm_classes, top_k=3):
    Xv = tfidf_vec.transform([text])
    probs = svm_model.predict_proba(Xv)[0]  # (n_classes,)

    top_idx = probs.argsort()[-top_k:][::-1]
    out = []
    for i in top_idx:
        cls_name = _svm_class_to_name(svm_classes[int(i)])
        out.append((cls_name, float(probs[int(i)])))
    return out

def svm_predict_safe_proba(text, svm_model, tfidf_vec, svm_classes,
                          top_k=3, min_proba=0.55, min_margin=0.15):
    topk = svm_predict_proba(text, svm_model, tfidf_vec, svm_classes, top_k=top_k)
    top1 = topk[0][1]
    top2 = topk[1][1] if len(topk) > 1 else 0.0
    margin = top1 - top2

    if (top1 < min_proba) or (margin < min_margin):
        return {"label": "uncertain", "confidence": float(top1), "top_k": topk}
    return {"label": topk[0][0], "confidence": float(top1), "top_k": topk}

# -------------------------
# Optional “final decision”: majority vote
# -------------------------
def majority_vote(roberta_safe, bert_safe, svm_safe):
    labels = [roberta_safe["label"], bert_safe["label"], svm_safe["label"]]
    labels_non_uncertain = [x for x in labels if x != "uncertain"]

    if not labels_non_uncertain:
        return {"final_label": "uncertain", "reason": "All models uncertain"}

    from collections import Counter
    c = Counter(labels_non_uncertain)
    top_label, top_count = c.most_common(1)[0]

    if top_count >= 2:
        return {"final_label": top_label, "reason": f"Majority vote ({top_count}/3)"}
    return {"final_label": "uncertain", "reason": "No majority agreement"}

# =========================
# 3) Streamlit UI
# =========================
st.set_page_config(page_title="Mental Health NLP - Comparator", layout="wide")
st.title("🧠 Mental Health Detection — RoBERTa vs BERT vs Calibrated SVM (names)")

device = get_device()
st.caption(f"Device: **{device}** (CPU is OK for inference)")

with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Top-K", 1, 5, 3)

    st.subheader("SAFE thresholds (all models)")
    min_proba = st.slider("Min probability", 0.10, 0.90, 0.55, 0.01)
    min_margin = st.slider("Min margin (top1-top2)", 0.00, 0.50, 0.15, 0.01)

    st.subheader("Display")
    show_final = st.checkbox("Show final decision (majority vote)", value=True)
    show_topk_tables = st.checkbox("Show Top-K table", value=True)
    show_bars = st.checkbox("Show bar chart", value=True)

with st.spinner("Loading models..."):
    roberta_tok, roberta_mdl, roberta_id2label = load_roberta(device)
    bert_tok, bert_mdl, bert_id2label = load_bert(device)
    svm_mdl, tfidf_vec, svm_classes = load_svm()

text = st.text_area(
    "Enter a text:",
    height=120,
    value="I feel anxious and overwhelmed with everything lately."
)

run = st.button("🔎 Predict")

if run:
    if not text.strip():
        st.warning("Please enter a non-empty text.")
        st.stop()

    # RoBERTa
    roberta_topk = predict_transformer(
        text, roberta_mdl, roberta_tok, roberta_id2label, device,
        top_k=top_k, id2label_keys_are_str=True
    )
    roberta_safe = predict_transformer_safe(
        text, roberta_mdl, roberta_tok, roberta_id2label, device,
        top_k=top_k, min_proba=min_proba, min_margin=min_margin,
        id2label_keys_are_str=True
    )

    # BERT
    bert_topk = predict_transformer(
        text, bert_mdl, bert_tok, bert_id2label, device,
        top_k=top_k, id2label_keys_are_str=False
    )
    bert_safe = predict_transformer_safe(
        text, bert_mdl, bert_tok, bert_id2label, device,
        top_k=top_k, min_proba=min_proba, min_margin=min_margin,
        id2label_keys_are_str=False
    )

    # SVM calibrated (names)
    svm_topk = svm_predict_proba(text, svm_mdl, tfidf_vec, svm_classes, top_k=top_k)
    svm_safe = svm_predict_safe_proba(
        text, svm_mdl, tfidf_vec, svm_classes,
        top_k=top_k, min_proba=min_proba, min_margin=min_margin
    )

    if show_final:
        final = majority_vote(roberta_safe, bert_safe, svm_safe)
        
        st.success(f"Final label: **{final['final_label']}** — {final['reason']}")

    st.divider()

    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("RoBERTa")
        st.markdown(f"**SAFE label:** `{roberta_safe['label']}`  \n**Confidence:** `{roberta_safe['confidence']:.4f}`")
        if show_topk_tables:
            st.dataframe(topk_to_table(roberta_topk, "probability"), use_container_width=True)
        if show_bars:
            st.bar_chart({lab: val for lab, val in roberta_topk})

    with c2:
        st.subheader("BERT")
        st.markdown(f"**SAFE label:** `{bert_safe['label']}`  \n**Confidence:** `{bert_safe['confidence']:.4f}`")
        if show_topk_tables:
            st.dataframe(topk_to_table(bert_topk, "probability"), use_container_width=True)
        if show_bars:
            st.bar_chart({lab: val for lab, val in bert_topk})

    with c3:
        st.subheader("SVM (TF-IDF) — Calibrated")
        st.markdown(f"**SAFE label:** `{svm_safe['label']}`  \n**Confidence:** `{svm_safe['confidence']:.4f}`")
        if show_topk_tables:
            st.dataframe(topk_to_table(svm_topk, "probability"), use_container_width=True)
        if show_bars:
            st.bar_chart({lab: val for lab, val in svm_topk})

    st.divider()
    st.info(
        "SAFE = returns `uncertain` when probability is low (< min_proba) "
        "or the model hesitates (top1-top2 < min_margin)."
    )
