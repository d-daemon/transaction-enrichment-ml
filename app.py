"""
app.py
---------------------------
Streamlit app for interactive transaction categorization.
"""

from io import StringIO

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from categorize_transactions import MIN_CONFIDENCE_THRESHOLD, predict_top_k
from src.brand_matcher import clean_merchant_name
from src.industry_classifier import classify_industry

# -------------------------------------------------------------------
# Streamlit Page Config
# -------------------------------------------------------------------
st.set_page_config(page_title="Transaction Categorization", layout="wide")
st.title("Banking Transaction Categorization Demo")

# -------------------------------------------------------------------
# Sidebar Controls
# -------------------------------------------------------------------
st.sidebar.header("Settings")
confidence_threshold = st.sidebar.slider(
    "Minimum confidence threshold for brand assignment",
    min_value=0.0,
    max_value=1.0,
    value=MIN_CONFIDENCE_THRESHOLD,
    step=0.05,
    help="Predictions below this confidence will be classified as 'Other'."
)

st.sidebar.markdown("---")
st.sidebar.write(f"**Current Threshold:** {confidence_threshold:.2f}")

# -------------------------------------------------------------------
# Free Text Brand Prediction with Top-3 Bar Chart
# -------------------------------------------------------------------
st.subheader("Try Free-Text Brand Prediction")
merchant_input = st.text_input(
    "Enter RAW_MERCHANT text", 
    placeholder="e.g. STARBUCKS #123"
)

if merchant_input:
    top_preds = predict_top_k(merchant_input, k=3)

    if not top_preds:
        st.warning("Model unavailable or empty input.")
    else:
        top_brand, top_conf = top_preds[0]

        if top_conf < confidence_threshold:
            st.warning(f"No confident match found (top={top_conf:.2%}).")
        else:
            st.success(f"Top Prediction: **{top_brand}** ({top_conf:.2%})")

        # Convert to DataFrame for plotting
        pred_df = pd.DataFrame(top_preds, columns=["Brand", "Confidence"])
        pred_df = pred_df.sort_values("Confidence", ascending=True)

        # Plot horizontal bar chart
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.barh(pred_df["Brand"], pred_df["Confidence"], color="steelblue")
        ax.set_xlabel("Confidence")
        ax.set_xlim(0, 1)
        for i, v in enumerate(pred_df["Confidence"]):
            ax.text(v + 0.01, i, f"{v:.2%}", va="center")
        st.pyplot(fig)

# -------------------------------------------------------------------
# Helper function to run categorization on uploaded CSV
# -------------------------------------------------------------------
def categorize_transactions(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    df["cleaned_merchant"] = df["RAW_MERCHANT"].apply(clean_merchant_name)

    def assign_brand(text):
        top_preds = predict_top_k(text, k=3)
        if not top_preds:
            return None
        brand, conf = top_preds[0]
        return brand if conf >= threshold else "Other"

    df["brand_pred"] = df["RAW_MERCHANT"].apply(assign_brand)

    industry_df = df.apply(
        lambda row: classify_industry(row["brand_pred"], row["MCC_CODE"]),
        axis=1,
        result_type="expand"
    )
    industry_df.columns = ["industry_t1_pred", "industry_t2_pred"]
    df = pd.concat([df, industry_df], axis=1)
    return df

# -------------------------------------------------------------------
# CSV Upload Section
# -------------------------------------------------------------------
st.write(
    "Upload a CSV file containing `RAW_MERCHANT` and `MCC_CODE` fields "
    "to categorize transactions by **brand** and **industry**."
)

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    # Read uploaded file
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    df_input = pd.read_csv(stringio)

    st.subheader("Preview of Uploaded Data")
    st.dataframe(df_input.head())

    required_cols = {"RAW_MERCHANT", "MCC_CODE"}
    if not required_cols.issubset(df_input.columns):
        st.error(f"CSV must contain columns: {required_cols}")
    else:
        with st.spinner("Categorizing transactions..."):
            df_output = categorize_transactions(df_input, confidence_threshold)

        st.success("Categorization complete.")
        st.subheader("Enriched Transactions")
        st.dataframe(df_output.head(20))

        csv = df_output.to_csv(index=False)
        st.download_button(
            label="Download Enriched CSV",
            data=csv,
            file_name="enriched_transactions.csv",
            mime="text/csv"
        )
else:
    st.info("Awaiting CSV file upload.")
