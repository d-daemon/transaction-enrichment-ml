"""
brand_matcher.py
----------------
Utility module for cleaning raw merchant names before
vectorization and brand classification.
"""

import re


def clean_merchant_name(name: str) -> str:
    """
    Normalize and clean raw merchant names for consistent
    vectorization and classification.

    - Lowercase
    - Remove branch IDs (#123)
    - Remove punctuation
    - Normalize whitespace
    """
    if not isinstance(name, str):
        return ""

    name = name.lower()
    name = re.sub(r'#\d+', '', name)         # remove branch identifiers
    name = re.sub(r'[^a-z0-9\s]', '', name)  # keep alphanumeric + space
    name = re.sub(r'\s+', ' ', name)         # normalize spaces
    return name.strip()
