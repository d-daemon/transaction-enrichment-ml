import pytest
from src.industry_classifier import classify_industry

def test_classify_by_brand():
    result = classify_industry("Starbucks", None)
    assert result == ("Food & Beverage", "Coffee Shops")

def test_classify_by_mcc():
    result = classify_industry(None, 5411)
    assert result == ("Retail", "Supermarkets")

def test_brand_precedence_over_mcc():
    result = classify_industry("Starbucks", 5411)
    # Brand mapping should take precedence
    assert result == ("Food & Beverage", "Coffee Shops")

def test_unknown_mcc_and_brand():
    result = classify_industry("UnknownBrand", 9999)
    assert result == (None, None)
