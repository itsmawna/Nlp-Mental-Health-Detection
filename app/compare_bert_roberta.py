import os
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =========================
# 1) Paths (change if needed)
# =========================
ROBERTA_DIR = r"C:\Users\21276\NLP-Mental-Health-Detection\models\roberta_export"
BERT_DIR    = r"C:\Users\21276\NLP-Mental-Health-Detection\models\bert_export"

# =========================
# 2) Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# 3) Load RoBERTa
# =========================
roberta_tokenizer = AutoTokenizer.from_pretrained(ROBERTA_DIR)
roberta_model = AutoModelForSequenceClassification.from_pretrained(ROBERTA_DIR).to(device)
roberta_model.eval()

# RoBERTa id2label from json (keys are strings)
with open(os.path.join(ROBERTA_DIR, "id2label.json"), "r", encoding="utf-8") as f:
    roberta_id2label = json.load(f)

# =========================
# 4) Load BERT
# =========================
bert_tokenizer = AutoTokenizer.from_pretrained(BERT_DIR)
bert_model = AutoModelForSequenceClassification.from_pretrained(BERT_DIR).to(device)
bert_model.eval()

# BERT id2label from config (keys are ints)
bert_id2label = {int(k): v for k, v in bert_model.config.id2label.items()}

# =========================
# 5) Generic predict functions
# =========================
def _predict_topk(text, model, tokenizer, id2label, top_k=3, max_len=160, id2label_keys_are_str=False):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_len
    )
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

def _predict_safe(text, model, tokenizer, id2label, top_k=3, max_len=160,
                  min_proba=0.55, min_margin=0.15, id2label_keys_are_str=False):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_len
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()

    top_idx = probs.argsort()[-top_k:][::-1]
    top1 = float(probs[int(top_idx[0])])
    top2 = float(probs[int(top_idx[1])]) if top_k > 1 else 0.0
    margin = top1 - top2

    topk = []
    for i in top_idx:
        key = str(int(i)) if id2label_keys_are_str else int(i)
        topk.append((id2label[key], float(probs[int(i)])))

    pred_label = topk[0][0]
    if (top1 < min_proba) or (margin < min_margin):
        return {"label": "uncertain", "confidence": top1, "top_k": topk}

    return {"label": pred_label, "confidence": top1, "top_k": topk}

# Wrappers
def roberta_predict(text, top_k=3, max_len=160):
    return _predict_topk(text, roberta_model, roberta_tokenizer, roberta_id2label,
                         top_k=top_k, max_len=max_len, id2label_keys_are_str=True)

def roberta_predict_safe(text, top_k=3, max_len=160, min_proba=0.55, min_margin=0.15):
    return _predict_safe(text, roberta_model, roberta_tokenizer, roberta_id2label,
                         top_k=top_k, max_len=max_len, min_proba=min_proba, min_margin=min_margin,
                         id2label_keys_are_str=True)

def bert_predict(text, top_k=3, max_len=160):
    return _predict_topk(text, bert_model, bert_tokenizer, bert_id2label,
                         top_k=top_k, max_len=max_len, id2label_keys_are_str=False)

def bert_predict_safe(text, top_k=3, max_len=160, min_proba=0.55, min_margin=0.15):
    return _predict_safe(text, bert_model, bert_tokenizer, bert_id2label,
                         top_k=top_k, max_len=max_len, min_proba=min_proba, min_margin=min_margin,
                         id2label_keys_are_str=False)

# =========================
# 6) Compare on a list of texts
# =========================
if __name__ == "__main__":
    test_texts = [
        "I feel anxious and overwhelmed with everything lately.",
        "I keep checking the door again and again even when I know it's locked.",
        "I have lost interest in everything and feel empty every day.",
        "I'm doing okay recently, sleeping well and focusing on my routine."
    ]

    print("\n============================")
    print("   RoBERTa vs BERT (Top-3)")
    print("============================")

    for t in test_texts:
        print("\nText:", t)

        r_top3 = roberta_predict(t, top_k=3)
        b_top3 = bert_predict(t, top_k=3)

        print("\nRoBERTa Top-3:", r_top3)
        print("BERT    Top-3:", b_top3)

        r_safe = roberta_predict_safe(t, top_k=3)
        b_safe = bert_predict_safe(t, top_k=3)

        print("\nRoBERTa SAFE:", r_safe)
        print("BERT    SAFE:", b_safe)
