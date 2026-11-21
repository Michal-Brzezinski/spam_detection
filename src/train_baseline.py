from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from preprocess import preprocess_text

# wczytanie danych
csv_path = Path(__file__).resolve().parent.parent / "data" / "spam.csv"

df = pd.read_csv(
    csv_path,
    encoding="latin1",
    usecols=[0, 1],
    names=["label", "text"],
    header=0,
)

# preprocessing
df["clean_text"] = df["text"].apply(preprocess_text)

# etykiety
df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

# Podział Train/Test
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# TF-IDF
vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# Model
model = LogisticRegression(
    random_state=42,
    max_iter=1000
)

model.fit(
    X_train_tfidf,
    y_train
)


# Predykcje
y_pred = model.predict(X_test_tfidf)


# Metryki
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"F1-score : {f1_score(y_test, y_pred):.4f}")