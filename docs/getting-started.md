# Getting Started

## Installation

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Data Preparation

```bash
uv run python -m scripts.generate_synthetic_data_with_labels
```

### Model Training

```bash
uv run python -m scripts.train_brand_classifier
```

### Batch Categorization Enrichment Pipeline

```bash
uv run python categorize_transactions.py \
    --input data/synthetic_raw_transactions.csv \
    --output output/enriched_transactions.csv
```

### Launch Dashboard

```bash
uv run streamlit run app.py
```

Open http://localhost:8501.

| [Home](index.md) | [Architecture →](architecture.md)
