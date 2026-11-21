from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from preprocess import preprocess_text

# ======================
# CONFIG & PATHS
# ======================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "spam.csv"
OUTPUT_PATH = BASE_DIR / "outputs"
CM_PATH = OUTPUT_PATH / "confusion_matrices"
CM_PATH.mkdir(parents=True, exist_ok=True)


# ======================
# HELPER FUNCTIONS
# ======================
def document_vector(doc: str, model: Word2Vec) -> np.ndarray:
    words = doc.split()
    vectors = [model.wv[word] for word in words if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)


def evaluate_and_plot(rep_name: str, model_name: str, model, X_tr, X_te, y_tr, y_te) -> dict:
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    cm = confusion_matrix(y_te, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Ham", "Spam"],  # czytelniejsze osie
        yticklabels=["Ham", "Spam"]
    )
    plt.title(f"{rep_name} + {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(CM_PATH / f"{rep_name}_{model_name}.png")
    plt.close()

    return {
        "Representation": rep_name,
        "Model": model_name,
        "Accuracy": round(float(accuracy_score(y_te, y_pred)), 4),
        "F1": round(float(f1_score(y_te, y_pred)), 4),
    }


# ======================
# DATA PREPARATION
# ======================
df = pd.read_csv(DATA_PATH, encoding="latin1", usecols=[0, 1], names=["label", "text"], header=0)
df["clean_text"] = df["text"].apply(preprocess_text)
df["label"] = df["label"].map({"ham": 0, "spam": 1})

X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)


# ======================
# REPRESENTATIONS
# ======================
# 1. Sparse (BOW & TF-IDF)
vectorizers = {
    "BOW": CountVectorizer(min_df=2),
    "TFIDF": TfidfVectorizer(ngram_range=(1, 2), min_df=2),
}

representations = {}
for name, vec in vectorizers.items():
    representations[name] = (vec.fit_transform(X_train), vec.transform(X_test))

# 2. Dense (Word2Vec)
train_tokens = [text.split() for text in X_train]
w2v_model = Word2Vec(sentences=train_tokens, vector_size=100, window=5, min_count=2, workers=4, seed=42)

representations["Word2Vec"] = (
    np.array([document_vector(doc, w2v_model) for doc in X_train]),
    np.array([document_vector(doc, w2v_model) for doc in X_test]),
)


# ======================
# TRAINING & EVALUATION
# ======================
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "LinearSVM": LinearSVC(random_state=42),
    "NaiveBayes": MultinomialNB(),
}

results = []
for rep_name, (X_tr, X_te) in representations.items():
    for model_name, model in models.items():
        # MultinomialNB nie obsługuje wartości ujemnych z Word2Vec
        if rep_name == "Word2Vec" and model_name == "NaiveBayes":
            continue

        res = evaluate_and_plot(rep_name, model_name, model, X_tr, X_te, y_train, y_test)
        results.append(res)


# ======================
# SAVE RESULTS
# ======================
results_df = pd.DataFrame(results).sort_values(by="F1", ascending=False)

print("\n===== RESULTS =====\n")
print(results_df.to_string(index=False))

results_df.to_csv(OUTPUT_PATH / "results.csv", index=False)
print(f"\nResults saved to {OUTPUT_PATH / 'results.csv'}")
