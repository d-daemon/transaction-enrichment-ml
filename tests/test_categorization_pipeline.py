import pandas as pd
from categorize_transactions import categorize_transactions

def test_pipeline_basic():
    df = pd.DataFrame({
        "TXN_ID": [1, 2],
        "RAW_MERCHANT": ["STARBUCKS #123", "McDonalds TST"],
        "MCC_CODE": [5814, 5814]
    })

    result = categorize_transactions(df.copy())

    assert "brand_pred" in result.columns
    assert "industry_t1_pred" in result.columns
    assert "industry_t2_pred" in result.columns

    assert result.loc[0, "brand_pred"].lower() == "starbucks"
    assert result.loc[0, "industry_t1_pred"] == "Food & Beverage"
    assert result.loc[0, "industry_t2_pred"] == "Coffee Shops"
