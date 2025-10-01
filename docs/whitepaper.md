# Machine Learning for Banking Transaction Enrichment: A Practical End-to-End Approach

---

## **1. Background**

Merchant names in banking transactions are often noisy, inconsistent, and contain unstructured location or terminal information. For example:

```markdown
"ST*RBCKS #123 HK"
"SHELL TST"
"MCDONALD LAKE JOSHUABURY"
```

Accurate brand and industry classification enables:

- Personal finance insights
- Transaction categorization
- Loyalty programs
- Compliance reporting

Traditional rules and fuzzy matching struggle with real-world messiness. Machine learning offers a scalable, maintainable solution.

---

## **2. Objective**

Build a **reproducible, open-source PoC** demonstrating:

1. **Data generation** and preparation  
2. **Brand classification** from noisy merchant text  
3. **Industry classification** using brand + MCC  
4. **Vectorization method comparison**  
5. **Interactive exploration** through a dashboard

---

## **3. End-to-End Process**

1. **Synthetic Data Generation**  
   - Generates merchant strings with realistic noise patterns and location suffixes  
   - Adds `"Other"` class to improve rejection handling

2. **Text Cleansing**  
   - Normalizes case, punctuation, and spacing  
   - Truncates suffix noise

3. **Vectorization & Model Training**  
   - Character n-gram TF-IDF + Logistic Regression  
   - Probability calibration (isotonic) for meaningful confidence scores

4. **Model Evaluation**  
   - Classification report and confusion matrix on hold-out set  
   - Vectorization method comparison

5. **Batch Categorization Pipeline**  
   - Enriches CSV files with `brand_pred`, `industry_t1_pred`, `industry_t2_pred`

6. **Interactive Dashboard**  
   - Free text prediction with top-3 bar chart  
   - CSV upload and enrichment  
   - Adjustable confidence threshold

---

## **4. Vectorization Methodology**

| Method             | Description               | Pros                  | Cons                 | Best For            |
| ------------------ | ------------------------- | --------------------- | -------------------- | ------------------- |
| Word TF-IDF        | Word token frequency      | Fast                  | Sensitive to typos   | Clean data          |
| Char n-gram TF-IDF | Character sequences (3–5) | Robust to noise       | Slightly opaque      | Noisy merchant text |
| fastText           | Dense subword embeddings  | Good for unseen words | Needs external model | Multilingual        |
| Transformers       | Contextual embeddings     | High accuracy         | Heavy                | R&D, large-scale    |

**Result:** Char n-gram TF-IDF achieves the best balance of accuracy, speed, and simplicity for this task.

---

## **5. Getting Started (For Other Users)**

### **Data Requirements**
| Field          | Type    | Example             |
| -------------- | ------- | ------------------- |
| `RAW_MERCHANT` | string  | `STARBUCKS #123 HK` |
| `MCC_CODE`     | integer | `5814`              |
| `AMOUNT`       | float   | `53.20`             |
| `CURRENCY`     | string  | `SGD`               |
| `CITY`         | string  | `Singapore`         |
| `COUNTRY`      | string  | `SG`                |

### **Input/Output per Step**

| Step           | Input                | Output                                                                 |
| -------------- | -------------------- | ---------------------------------------------------------------------- |
| Synthetic Data | —                    | `synthetic_raw_transactions.csv`, `brand_training.csv`                 |
| Training       | `brand_training.csv` | `models/brand_classifier.joblib`                                       |
| Enrichment     | Raw CSV              | Enriched CSV with `brand_pred`, `industry_t1_pred`, `industry_t2_pred` |
| Dashboard      | User text or CSV     | Top-3 predictions or enriched DataFrame                                |

---

## **6. Results**

- Char n-gram TF-IDF model achieves >95% accuracy on clean synthetic hold-out data  
- Handles typical real-world distortions:
  - `MCDONALD` vs `MCDONALDS`  
  - `ST*RBCKS #123` vs `STARBUCKS`  
- “Other” class correctly captures unseen merchants

---

## **7. Extending the Solution**

- Replace synthetic data with real merchant data  
- Integrate embeddings for semantic similarity  
- Deploy model as an API  
- Connect dashboard to live transaction streams

---

## **8. Conclusion**

This PoC demonstrates a **realistic, production-aligned** approach to transaction enrichment using modern ML techniques.  
It balances robustness, interpretability, and deployment simplicity — making it ideal for both educational purposes and as a foundation for real banking use cases.

---

## **Appendix: References**

- scikit-learn Probability Calibration: [https://scikit-learn.org/stable/modules/calibration.html](https://scikit-learn.org/stable/modules/calibration.html)  
- Character n-gram TF-IDF for noisy text: [https://research.google/pubs/pub37842/](https://research.google/pubs/pub37842/)  
- MCC Codes: ISO 18245

---
