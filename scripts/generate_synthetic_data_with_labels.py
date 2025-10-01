"""
generate_synthetic_data_with_labels.py
-------------------------
Generate synthetic transaction data with brand labels for training.
"""

import random
from datetime import datetime
from pathlib import Path

import pandas as pd
from faker import Faker

from src.brand_matcher import clean_merchant_name

fake = Faker()

# -------------------------------------------------------------------
# Reference data
# -------------------------------------------------------------------
BRANDS = [
    {"brand": "Starbucks", "mcc": 5814},
    {"brand": "McDonalds", "mcc": 5814},
    {"brand": "FairPrice", "mcc": 5411},
    {"brand": "Grab", "mcc": 4121},
    {"brand": "Shell", "mcc": 5541},
    {"brand": "Apple Store", "mcc": 5732},
    {"brand": "Guardian", "mcc": 5912},
]

CITY_COUNTRY_CURRENCY = [
    {"city": "Singapore", "country": "SG", "currency": "SGD"},
    {"city": "Hong Kong", "country": "HK", "currency": "HKD"},
    {"city": "Kuala Lumpur", "country": "MY", "currency": "MYR"},
    {"city": "Bangkok", "country": "TH", "currency": "THB"},
    {"city": "Tokyo", "country": "JP", "currency": "JPY"},
    {"city": "Sydney", "country": "AU", "currency": "AUD"},
    {"city": "London", "country": "GB", "currency": "GBP"},
]

# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------
def generate_other_examples(n=3000, seed=42):
    """
    Generate synthetic 'Other' merchant examples for training.
    These represent unknown / noise merchants to help the model
    learn a reject class.
    """
    random.seed(seed)
    Faker.seed(seed)

    others = []
    for _ in range(n):
        noise = fake.company()[:15]  # trim long names
        cleaned = clean_merchant_name(noise)
        others.append({"cleaned": cleaned, "BRAND": "Other"})
    return pd.DataFrame(others)

def generate_raw_merchant(brand_name: str) -> str:
    """
    Generate noisy merchant strings that mimic real transaction data.
    """
    patterns = [
        lambda b: f"{b.upper()} #{random.randint(1,999)}",
        lambda b: f"{b} {fake.city()}",
        lambda b: f"{b[:4]}*{b[4:]}",
        lambda b: f"{b.upper()}-{random.choice(['Mall', 'TST', 'HQ'])}",
        lambda b: b.upper(),
    ]
    return random.choice(patterns)(brand_name)

def generate_transaction(txn_id: int) -> dict:
    """
    Generate a single synthetic transaction record, including brand label for training.
    """
    brand_info = random.choice(BRANDS)
    merchant_name = generate_raw_merchant(brand_info["brand"])
    location = random.choice(CITY_COUNTRY_CURRENCY)

    return {
        "TXN_ID": txn_id,
        "RAW_MERCHANT": merchant_name,
        "MCC_CODE": brand_info["mcc"],
        "AMOUNT": round(random.uniform(1, 2000), 2),
        "CURRENCY": location["currency"],
        "TIMESTAMP": fake.date_time_between(start_date="-6M", end_date="now").strftime("%Y-%m-%d %H:%M:%S"),
        "CITY": location["city"],
        "COUNTRY": location["country"],
        "BRAND": brand_info["brand"],  # used only for training dataset
    }

def generate_datasets(n: int = 5000, seed: int = 42):
    """
    Generate both raw transactions and labeled training dataset.
    """
    random.seed(seed)
    Faker.seed(seed)

    records = [generate_transaction(i + 1) for i in range(n)]
    df_full = pd.DataFrame(records)

    # Raw transactions (no labels)
    df_raw = df_full.drop(columns=["BRAND"])

    # Training dataset: cleaned merchant + brand label
    df_train = pd.DataFrame({
        "cleaned": df_full["RAW_MERCHANT"].apply(clean_merchant_name),
        "BRAND": df_full["BRAND"]
    })

    return df_raw, df_train

# -------------------------------------------------------------------
# Main script
# -------------------------------------------------------------------
if __name__ == "__main__":
    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate raw + labeled brand data
    df_raw, df_train = generate_datasets(5000)

    # Generate 'Other' examples for reject class
    df_other = generate_other_examples(n=3000)
    df_train = pd.concat([df_train, df_other], ignore_index=True)

    # Save outputs
    raw_path = output_dir / "synthetic_raw_transactions.csv"
    train_path = output_dir / "brand_training.csv"

    df_raw.to_csv(raw_path, index=False)
    df_train.to_csv(train_path, index=False)

    print(f"Saved raw transaction data to: {raw_path}")
    print(f"Saved brand training data to: {train_path}")
    print("\nSample training data:")
    print(df_train.sample(10, random_state=42))
