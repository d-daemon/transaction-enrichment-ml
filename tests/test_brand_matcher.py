import pytest
from src.brand_matcher import clean_merchant_name, match_brand

def test_clean_merchant_name_basic():
    assert clean_merchant_name("STARBUCKS #123!") == "starbucks"

def test_clean_merchant_name_spacing():
    assert clean_merchant_name("  McDonalds   TST  ") == "mcdonalds tst"

def test_match_brand_exact():
    assert match_brand("starbucks") == "starbucks"

def test_match_brand_fuzzy():
    # Should match "starbucks" despite missing letters
    assert match_brand("starbks") == "starbucks"

def test_match_brand_none():
    assert match_brand("randomshop") is None
