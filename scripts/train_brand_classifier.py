"""
train_brand_classifier.py
-------------------------
Train a brand classification model using character n-gram TF-IDF
and probability calibration (isotonic) for better confidence scores.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

DATA_PATH = Path("data/brand_training.csv")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "brand_classifier.joblib"


def load_training_data():
    """Load labeled training data for brand classification."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found at {DATA_PATH}. "
            "Run `uv run python -m scripts.generate_synthetic_data_with_labels` first."
        )
    df = pd.read_csv(DATA_PATH)
    if "cleaned" not in df.columns or "BRAND" not in df.columns:
        raise ValueError("Training data must contain 'cleaned' and 'BRAND' columns.")
    return df


def train_model(df: pd.DataFrame):
    """Train character n-gram TF-IDF + calibrated logistic regression model."""
    X = df["cleaned"].astype(str)
    y = df["BRAND"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Character n-gram vectorizer (robust to typos)
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5))

    # Base classifier
    base_clf = LogisticRegression(max_iter=1000, multi_class="auto")

    # Probability calibration wrapper (5-fold isotonic)
    calibrated_clf = CalibratedClassifierCV(base_clf, cv=5, method="isotonic")

    model = Pipeline([("tfidf", vectorizer), ("calibrated_clf", calibrated_clf)])

    print("Training model...")
    model.fit(X_train, y_train)

    # Evaluate
    print("\nEvaluating model on hold-out set...")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, zero_division=0))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model


def save_model(model):
    """Persist trained model to disk."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    df = load_training_data()
    model = train_model(df)
    save_model(model)
