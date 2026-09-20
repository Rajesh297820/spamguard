"""Train the SpamGuard NLP spam classifier.

The script downloads the public UCI SMS Spam Collection when data/spam.csv
is not present, converts it to a simple CSV, trains a TF-IDF + Logistic
Regression model, evaluates it, and saves the trained artifacts.
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_FILE = DATA_DIR / "spam.csv"
MODEL_FILE = BASE_DIR / "spam_classifier.pkl"
VECTORIZER_FILE = BASE_DIR / "vectorizer.pkl"

# UCI's current static download URL, with the older UCI path as a fallback.
DATASET_URLS = [
    "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip",
]


def clean_text(text: str) -> str:
    """Normalize text while retaining useful spam-related tokens."""
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " urltoken ", text)
    text = re.sub(r"\S+@\S+", " emailtoken ", text)
    text = re.sub(r"\d+", " numbertoken ", text)
    text = re.sub(r"[^a-z0-9_\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def download_dataset() -> None:
    """Download and convert the UCI dataset to data/spam.csv."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None

    print("Dataset not found. Downloading the UCI SMS Spam Collection...")

    for url in DATASET_URLS:
        try:
            request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(request, timeout=30) as response:
                archive_bytes = response.read()

            with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
                member = next(
                    (name for name in archive.namelist() if name.endswith("SMSSpamCollection")),
                    None,
                )
                if member is None:
                    raise FileNotFoundError("SMSSpamCollection was not found in the downloaded ZIP.")

                with archive.open(member) as raw_file:
                    df = pd.read_csv(
                        raw_file,
                        sep="\t",
                        header=None,
                        names=["label", "message"],
                        encoding="utf-8",
                    )

            df = df.dropna(subset=["label", "message"])
            df["label"] = df["label"].astype(str).str.strip().str.lower()
            df["message"] = df["message"].astype(str).str.strip()
            df = df[df["label"].isin(["ham", "spam"])]
            df = df.drop_duplicates(subset=["label", "message"]).reset_index(drop=True)
            df.to_csv(DATA_FILE, index=False, encoding="utf-8")

            print(f"Dataset saved to: {DATA_FILE}")
            print(f"Rows available after cleaning: {len(df):,}")
            return
        except Exception as exc:  # noqa: BLE001 - provide a useful fallback error
            last_error = exc

    raise RuntimeError(
        "Could not download the dataset automatically. "
        "Download the UCI SMS Spam Collection and save it as data/spam.csv, "
        f"then run this script again. Last error: {last_error}"
    )


def load_dataset() -> pd.DataFrame:
    """Load an existing CSV, accepting common SMS spam column names."""
    if not DATA_FILE.exists():
        download_dataset()

    try:
        df = pd.read_csv(DATA_FILE, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(DATA_FILE, encoding="latin-1")

    # Support both our generated CSV and common Kaggle-style spam.csv files.
    normalized = {str(col).strip().lower(): col for col in df.columns}

    label_col = next(
        (normalized[name] for name in ("label", "v1", "class") if name in normalized),
        None,
    )
    message_col = next(
        (normalized[name] for name in ("message", "v2", "text") if name in normalized),
        None,
    )

    if label_col is None or message_col is None:
        if len(df.columns) >= 2:
            label_col, message_col = df.columns[:2]
        else:
            raise ValueError("spam.csv must contain at least two columns: label and message.")

    data = df[[label_col, message_col]].copy()
    data.columns = ["label", "message"]
    data["label"] = data["label"].astype(str).str.strip().str.lower()
    data["message"] = data["message"].astype(str).str.strip()
    data = data[data["label"].isin(["ham", "spam"])]
    data = data[data["message"].str.len() > 0]
    data = data.drop_duplicates(subset=["label", "message"]).reset_index(drop=True)

    if len(data) < 100:
        raise ValueError(
            "The dataset contains too few valid rows. Please use the full UCI SMS Spam Collection."
        )

    return data


def main() -> None:
    data = load_dataset()

    print("\n========== SpamGuard Training ==========")
    print(f"Total messages : {len(data):,}")
    print(f"Ham messages   : {(data['label'] == 'ham').sum():,}")
    print(f"Spam messages  : {(data['label'] == 'spam').sum():,}")

    X = data["message"].map(clean_text)
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
        max_features=20_000,
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    classifier = LogisticRegression(
        max_iter=1_000,
        random_state=42,
        class_weight="balanced",
    )
    classifier.fit(X_train_tfidf, y_train)

    predictions = classifier.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, pos_label="spam", zero_division=0)
    recall = recall_score(y_test, predictions, pos_label="spam", zero_division=0)
    f1 = f1_score(y_test, predictions, pos_label="spam", zero_division=0)
    matrix = confusion_matrix(y_test, predictions, labels=["ham", "spam"])

    print("\n========== Evaluation ==========")
    print(f"Accuracy        : {accuracy * 100:.2f}%")
    print(f"Spam Precision  : {precision * 100:.2f}%")
    print(f"Spam Recall     : {recall * 100:.2f}%")
    print(f"Spam F1 Score   : {f1 * 100:.2f}%")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, target_names=["ham", "spam"], zero_division=0))
    print("Confusion matrix [ham, spam]:")
    print(matrix)

    joblib.dump(classifier, MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)

    print("\n========== Files Created ==========")
    print(f"Model       : {MODEL_FILE.name}")
    print(f"Vectorizer  : {VECTORIZER_FILE.name}")
    print("Training completed successfully.")


if __name__ == "__main__":
    main()
