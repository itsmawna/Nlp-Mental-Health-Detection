# =========================
# Complete inference code (load + predict)
# Works with your exported Kaggle folder: model/roberta_export/
# =========================

import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_DIR = r"C:\Users\21276\NLP-Mental-Health-Detection\models\roberta_export"  

# ---- 2) Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---- 3) Load tokenizer + model
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
model.eval()

# ---- 4) Load label mapping (from your export)
with open(os.path.join(MODEL_DIR, "id2label.json"), "r", encoding="utf-8") as f:
    id2label = json.load(f)  # keys are strings: "0","1",...

# (optional) load label2id if needed later
label2id_path = os.path.join(MODEL_DIR, "label2id.json")
label2id = None
if os.path.exists(label2id_path):
    with open(label2id_path, "r", encoding="utf-8") as f:
        label2id = json.load(f)

# ---- 5) Prediction function (Top-K)
def predict(text: str, top_k: int = 3, max_len: int = 160):
    """
    Returns: list of (label, probability) for the top_k classes
    """
    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_len
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Forward
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()

    # Top-k
    top_idx = probs.argsort()[-top_k:][::-1]
    return [(id2label[str(int(i))], float(probs[int(i)])) for i in top_idx]

# ---- 6) Optional "uncertain" mode (recommended for real apps)
def predict_safe(
    text: str,
    top_k: int = 3,
    max_len: int = 160,
    min_proba: float = 0.55,
    min_margin: float = 0.15
):
    """
    If the model isn't confident, returns label="uncertain".
    """
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

    topk = [(id2label[str(int(i))], float(probs[int(i)])) for i in top_idx]
    pred_label = topk[0][0]

    if (top1 < min_proba) or (margin < min_margin):
        return {"label": "uncertain", "confidence": top1, "top_k": topk}

    return {"label": pred_label, "confidence": top1, "top_k": topk}

# ---- 7) Quick test
if __name__ == "__main__":
    test_texts = [
        "I feel anxious and overwhelmed with everything lately.",
        "I keep checking the door again and again even when I know it's locked.",
        "I have lost interest in everything and feel empty every day.",
        "I'm doing okay recently, sleeping well and focusing on my routine."
    ]

    print("\n--- PREDICT (Top-3) ---")
    for t in test_texts:
        print("\nText:", t)
        print("Top-3:", predict(t, top_k=3))

    print("\n--- PREDICT_SAFE (with 'uncertain') ---")
    for t in test_texts:
        out = predict_safe(t, top_k=3)
        print("\nText:", t)
        print(out)
