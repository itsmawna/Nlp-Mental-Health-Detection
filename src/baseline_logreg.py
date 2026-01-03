# =========================
# Baseline: TF-IDF + Logistic Regression
# =========================

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# =========================
# 1) Config
# =========================
SEED = 42
DATA_PATH = r"C:\Users\21276\NLP-Mental-Health-Detection\data\processed\thefinal_dataset.csv"  
TEXT_COLUMN = "text"                               
LABEL_COLUMN = "status"

OUTPUT_DIR = "models/baseline_lr"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# 2) Load data
# =========================
print("\nLoading data...")
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str).fillna("")
df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(str).str.lower().str.strip()

# Label encoding
labels = sorted(df[LABEL_COLUMN].unique())
label_map = {lab: i for i, lab in enumerate(labels)}
id_to_label = {i: lab for lab, i in label_map.items()}
df["label_id"] = df[LABEL_COLUMN].map(label_map)

X = df[TEXT_COLUMN]
y = df["label_id"]

print("Classes:", labels)

# =========================
# 3) Train / Test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=SEED,
    stratify=y
)

print("Train:", len(X_train), "Test:", len(X_test))

# =========================
# 4) TF-IDF Vectorization
# =========================
vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    min_df=3,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF shape:", X_train_tfidf.shape)

# =========================
# 5) Logistic Regression (IMBALANCE FIX)
# =========================
model = LogisticRegression(
    solver="saga",
    max_iter=3000,
    n_jobs=-1,
    class_weight="balanced",   # imbalance handled here
    random_state=SEED
)

model.fit(X_train_tfidf, y_train)

# =========================
# 6) Evaluation
# =========================
y_pred = model.predict(X_test_tfidf)

print("\n=== TEST RESULTS (Baseline) ===")
print(classification_report(
    y_test,
    y_pred,
    target_names=[id_to_label[i] for i in sorted(id_to_label)],
    zero_division=0
))

f1_macro = f1_score(y_test, y_pred, average="macro")
print("Macro F1:", f1_macro)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix – Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
plt.show()

# =========================
# 7) Export model (for interface)
# =========================
joblib.dump(model, os.path.join(OUTPUT_DIR, "logreg_model.joblib"))
joblib.dump(vectorizer, os.path.join(OUTPUT_DIR, "tfidf_vectorizer.joblib"))
joblib.dump(label_map, os.path.join(OUTPUT_DIR, "label_map.joblib"))
joblib.dump(id_to_label, os.path.join(OUTPUT_DIR, "id_to_label.joblib"))

print("\nBaseline model exported to:", OUTPUT_DIR)

