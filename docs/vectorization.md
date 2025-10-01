# Vectorization Methods

| Method | Pros | Cons | Best For |
|-------|------|------|---------|
| Word TF-IDF | Simple | Sensitive to typos | Clean data |
| **Char n-gram TF-IDF** | Robust, fast | Slightly opaque | Noisy merchant text ✅ |
| fastText | Semantic power | External dependency | Multilingual |
| Transformers | High accuracy | Heavy | R&D only |

**Chosen Method:** Character n-gram TF-IDF (3–5) + Logistic Regression + Isotonic calibration.

👉 Next: [Resources →](resources.md)
