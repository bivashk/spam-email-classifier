import os
import io
import zipfile
import requests
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

UCI_ZIP_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
RAW_FILE = os.path.join(DATA_DIR, "SMSSpamCollection")
MODEL_PATH = os.path.join(MODELS_DIR, "spam_model.joblib")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.txt")

def download_dataset():
    if os.path.exists(RAW_FILE):
        print("[i] Dataset already present at", RAW_FILE)
        return RAW_FILE
    print("[i] Downloading dataset from UCI...")
    r = requests.get(UCI_ZIP_URL, timeout=30)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        # The file inside the zip is named 'SMSSpamCollection'
        z.extractall(DATA_DIR)
    if not os.path.exists(RAW_FILE):
        raise FileNotFoundError("Expected SMSSpamCollection after extraction, but not found.")
    print("[i] Downloaded and extracted to", RAW_FILE)
    return RAW_FILE

def load_data(path):
    # The file is tab-separated with two columns: label \t text
    df = pd.read_csv(path, sep="\t", header=None, names=["label", "text"])
    # Keep only non-empty text rows
    df = df.dropna(subset=["text"]).reset_index(drop=True)
    print(f"[i] Loaded {len(df)} rows. Spam: {(df['label']=='spam').sum()}, Ham: {(df['label']=='ham').sum()}")
    return df

def build_pipelines(max_features=20000):
    tfidf = TfidfVectorizer(stop_words="english", lowercase=True, max_features=max_features)
    pipe_nb = Pipeline([("tfidf", tfidf), ("clf", MultinomialNB())])
    pipe_lr = Pipeline([("tfidf", tfidf), ("clf", LogisticRegression(max_iter=1000))])
    return {"naive_bayes": pipe_nb, "log_reg": pipe_lr}

def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, preds, average="weighted", zero_division=0)
    report = classification_report(y_test, preds, zero_division=0)
    return acc, precision, recall, f1, report

def main():
    path = download_dataset()
    df = load_data(path)

    X = df["text"].astype(str).values
    y = df["label"].astype(str).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipelines = build_pipelines()
    scores = {}

    best_name, best_model, best_f1 = None, None, -1.0

    for name, pipe in pipelines.items():
        print(f"[i] Training {name} ...")
        pipe.fit(X_train, y_train)
        acc, precision, recall, f1, report = evaluate(pipe, X_test, y_test)
        scores[name] = {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "report": report}
        print(f"[i] {name} -> acc={acc:.4f} f1={f1:.4f}")
        if f1 > best_f1:
            best_name, best_model, best_f1 = name, pipe, f1

    # Save best model
    joblib.dump(best_model, MODEL_PATH)
    print(f"[i] Saved best model ({best_name}) to {MODEL_PATH}")

    # Save metrics
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        for name, m in scores.items():
            f.write(f"### {name}\n")
            f.write(f"accuracy: {m['accuracy']:.4f}\nprecision: {m['precision']:.4f}\nrecall: {m['recall']:.4f}\nf1: {m['f1']:.4f}\n\n")
            f.write(m["report"] + "\n\n")
        f.write(f"Best model: {best_name} (f1={best_f1:.4f})\n")

    print("[i] Metrics written to", METRICS_PATH)

if __name__ == "__main__":
    main()
