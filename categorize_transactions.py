"""
categorize_transactions.py
---------------------------
Pipeline to enrich raw transaction data with brand and industry
based on merchant text and MCC codes.

Expected input columns:
- TXN_ID
- RAW_MERCHANT
- MCC_CODE
- AMOUNT
- CURRENCY
- TIMESTAMP
- CITY
- COUNTRY
"""

import argparse
import warnings
from pathlib import Path

import joblib
import pandas as pd

from src.brand_matcher import clean_merchant_name
from src.industry_classifier import classify_industry

MODEL_PATH = Path("models/brand_classifier.joblib")
MIN_CONFIDENCE_THRESHOLD = 0.25

# -------------------------------------------------------------------
# Model loading and prediction utilities
# -------------------------------------------------------------------

_brand_model = None  # cache

def load_brand_model():
    global _brand_model
    if _brand_model is None:
        if not MODEL_PATH.exists():
            warnings.warn(
                f"Brand classifier model not found at {MODEL_PATH}. "
                f"Run `uv run python -m scripts.train_brand_classifier` first to train it."
            )
            return None
        _brand_model = joblib.load(MODEL_PATH)
    return _brand_model

def predict_brand(merchant_text: str):
    """Predict the single top brand from merchant text."""
    if not isinstance(merchant_text, str) or merchant_text.strip() == "":
        return None

    model = load_brand_model()
    if model is None:
        return None

    cleaned = clean_merchant_name(merchant_text)
    return model.predict([cleaned])[0]

def predict_brand_with_confidence(merchant_text: str):
    """Return (brand, confidence) for the top predicted brand."""
    if not isinstance(merchant_text, str) or merchant_text.strip() == "":
        return None, None

    model = load_brand_model()
    if model is None:
        return None, None

    cleaned = clean_merchant_name(merchant_text)
    proba = model.predict_proba([cleaned])[0]
    classes = model.classes_
    top_idx = proba.argmax()
    return classes[top_idx], proba[top_idx]

def predict_top_k(merchant_text: str, k: int = 3):
    """Return top-k (brand, confidence) predictions sorted by confidence."""
    if not isinstance(merchant_text, str) or merchant_text.strip() == "":
        return []

    model = load_brand_model()
    if model is None:
        return []

    cleaned = clean_merchant_name(merchant_text)
    proba = model.predict_proba([cleaned])[0]
    classes = model.classes_
    top_indices = proba.argsort()[::-1][:k]
    return [(classes[i], proba[i]) for i in top_indices]

# -------------------------------------------------------------------
# Categorization pipeline
# -------------------------------------------------------------------

def categorize_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply text cleansing, brand assignment, and industry classification
    to a transactions DataFrame. Location and currency fields are passed through.
    """
    # Step 1: Cleanse merchant names
    df["cleaned_merchant"] = df["RAW_MERCHANT"].apply(clean_merchant_name)

    # Step 2: Brand assignment using ML model
    df["brand_pred"] = df["cleaned_merchant"].apply(predict_brand)

    # Step 3: Industry classification (brand first, then MCC fallback)
    industry_df = df.apply(
        lambda row: classify_industry(row["brand_pred"], row["MCC_CODE"]),
        axis=1,
        result_type="expand"
    )
    industry_df.columns = ["industry_t1_pred", "industry_t2_pred"]

    # Step 4: Combine results
    df = pd.concat([df, industry_df], axis=1)

    # Reorder columns for readability
    column_order = [
        "TXN_ID",
        "RAW_MERCHANT",
        "cleaned_merchant",
        "brand_pred",
        "industry_t1_pred",
        "industry_t2_pred",
        "MCC_CODE",
        "AMOUNT",
        "CURRENCY",
        "TIMESTAMP",
        "CITY",
        "COUNTRY"
    ]
    df = df[[col for col in column_order if col in df.columns]]
    return df

# -------------------------------------------------------------------
# CLI entrypoint
# -------------------------------------------------------------------

def main(input_path: str, output_path: str):
    input_file = Path(input_path)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading raw transactions from: {input_file}")
    df = pd.read_csv(input_file)

    print("Running categorization pipeline...")
    enriched_df = categorize_transactions(df)

    print(f"Saving enriched data to: {output_file}")
    enriched_df.to_csv(output_file, index=False)

    print("Categorization complete.")
    print(enriched_df.head(10))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enrich raw transactions with brand and industry")
    parser.add_argument("--input", required=True, help="Path to input raw transactions CSV")
    parser.add_argument("--output", required=True, help="Path to output enriched CSV")
    args = parser.parse_args()
    main(args.input, args.output)
