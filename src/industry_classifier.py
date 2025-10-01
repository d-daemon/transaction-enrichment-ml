"""
industry_classifier.py
-----------------------
Module for mapping brands and MCC codes to industry Tier 1 and Tier 2 categories.
"""

# Example MCC lookup table
MCC_LOOKUP = {
    5814: ("Food & Beverage", "Coffee Shops"),
    5411: ("Retail", "Supermarkets"),
    5541: ("Transport", "Fuel"),
    5732: ("Retail", "Electronics"),
    5912: ("Retail", "Pharmacy"),
    4121: ("Transport", "Ride Hailing"),
}

# Optional brand-specific overrides for industry classification
BRAND_INDUSTRY_MAP = {
    "starbucks": ("Food & Beverage", "Coffee Shops"),
    "mcdonalds": ("Food & Beverage", "Fast Food"),
    "fairprice": ("Retail", "Supermarkets"),
    "grab": ("Transport", "Ride Hailing"),
    "shell": ("Transport", "Fuel"),
}


def classify_industry(brand: str | None, mcc_code: int | str | None) -> tuple[str | None, str | None]:
    """
    Determine Industry Tier 1 and Tier 2 based on brand or MCC code.
    Brand takes precedence; falls back to MCC mapping if brand is unknown or 'Other'.
    """
    # Brand mapping takes precedence
    if isinstance(brand, str):
        b = brand.lower().strip()
        if b != "other" and b in BRAND_INDUSTRY_MAP:
            return BRAND_INDUSTRY_MAP[b]

    # Fallback to MCC mapping
    try:
        mcc_int = int(mcc_code)
        if mcc_int in MCC_LOOKUP:
            return MCC_LOOKUP[mcc_int]
    except (TypeError, ValueError):
        pass

    return (None, None)
