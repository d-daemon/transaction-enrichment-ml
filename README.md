# Transaction Enrichment & Brand Classification PoC

<!-- Project Metadata -->
![Python Version](https://img.shields.io/badge/python-3.13-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux-lightgrey)

<!-- Repo Stats -->
![Last Commit](https://img.shields.io/github/last-commit/d-daemon/transaction-enrichment-ml)
![Repo Stars](https://img.shields.io/github/stars/d-daemon/transaction-enrichment-ml?style=social)
![Repo Forks](https://img.shields.io/github/forks/d-daemon/transaction-enrichment-ml?style=social)

This repository demonstrates a complete **end-to-end machine learning pipeline** for enriching banking transactions by:

- Assigning **brand labels** from noisy merchant strings
- Classifying transactions into **industry hierarchies**
- Providing **interactive visualization** and exploration through a Streamlit dashboard

The project is designed to highlight advanced ML skills, data engineering practices, and practical production-style techniques — making it an excellent PoC for portfolio and hiring showcases.

---

## Features

- **Synthetic data generator** for merchant + transaction data  
- **Text cleaning & normalization** for real-world merchant strings  
- **Brand classification** using calibrated ML models (character n-gram TF-IDF + Logistic Regression)  
- **Industry classification** using brand + MCC code  
- **Vectorization strategy comparison** (word, char n-gram, embeddings, transformers)  
- **Streamlit dashboard** with:
  - Free-text top-K brand prediction
  - Interactive confidence threshold control
  - Batch CSV enrichment with download

---

## Project Structure

```
txn_enrichment/
├── data/   # Synthetic data and training datasets
├── models/ # Saved ML models
├── output/ # Enriched CSV outputs
├── scripts/
│ ├── generate_synthetic_data_with_labels.py
│ ├── train_brand_classifier.py
├── src/
│ ├── brand_matcher.py
│ ├── industry_classifier.py
├── categorize_transactions.py # Batch categorization pipeline
├── app.py # Streamlit dashboard
└── README.md
```

---

## Quick Start

### 1. Install Dependencies

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 2. Generate Synthetic Data

```bash
uv run python -m scripts.generate_synthetic_data_with_labels
```

### 3. Train the Brand Classifier

```bash
uv run python -m scripts.train_brand_classifier
```

### 4. Initiate Batch Categorization Pipeline

```bash
uv run python categorize_transactions.py \
    --input data/synthetic_raw_transactions.csv \
    --output output/enriched_transactions.csv
```

### 5. Launch the Dashboard

```bash
uv run streamlit run app.py
```

Open http://localhost:8501 in your browser.

## Model

- **Vectorization**: Character n-gram TF-IDF (3–5)
- **Classifier**: Logistic Regression + Isotonic probability calibration
- **Advantages**:
  - Robust to typos, truncation, and noisy formatting
  - Lightweight and fast inference
  - Excellent baseline for production merchant classification

## Example

| RAW_MERCHANT             | brand_pred | industry_t1_pred | industry_t2_pred |
| ------------------------ | ---------- | ---------------- | ---------------- |
| STARBUCKS #123 HK        | Starbucks  | F&B              | Coffee Chains    |
| SHELL TST                | Shell      | Fuel             | Gas Stations     |
| MCDONALD LAKE JOSHUABURY | McDonalds  | F&B              | Fast Food        |
| FUSION INTL GROUP        | Other      | Unknown          | Unknown          |

## Vectorization Comparison

| Vector Method       | Pros                                      | Cons                                   | Accuracy                    |
| ------------------- | ----------------------------------------- | -------------------------------------- | --------------------------- |
| Word TF-IDF         | Fast, interpretable                       | Brittle to noise                       | Medium                      |
| Char n-gram TF-IDF  | Robust to messy text, lightweight         | Slightly less interpretable            | High ✅                      |
| fastText Embeddings | Semantic generalization                   | Requires external model, slower        | High                        |
| Transformers        | State-of-the-art accuracy, semantic power | Heavy, slower inference, overkill here | Very High (but impractical) |

## Tech Stack

- **Python 3.13** + **uv** for fast package management
- **scikit-learn** for ML models and calibration
- **Streamlit** for dashboard UI
- **Faker** for synthetic data generation
- **pandas** for data processing

## Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/new-feature`)
3. **Make** your changes
4. **Submit** a pull request

## License

This project is licensed under the **MIT License**.

---
