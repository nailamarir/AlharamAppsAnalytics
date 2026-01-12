"""
Username cleaning and preprocessing utilities.

Handles various username formats including:
- Arabic names
- Latin names
- Arabizi (Arabic written with Latin letters and numbers)
- Mixed scripts
"""

import re
import unicodedata
from typing import Optional

import pandas as pd


# Arabizi digit-to-letter mapping
ARABIZI_MAP = {
    "2": "a",   # hamza
    "3": "a",   # ain
    "5": "kh",  # kha
    "6": "t",   # ta marbuta
    "7": "h",   # ha
    "8": "gh",  # ghain
    "9": "s",   # sad
}


def _is_punct_or_symbol(ch: str) -> bool:
    """Check if character is punctuation or symbol."""
    cat = unicodedata.category(ch)
    return cat.startswith("P") or cat.startswith("S")


def clean_username(name: Optional[str], min_letters: int = 3) -> str:
    """
    Clean and normalize a username string.

    Args:
        name: Raw username string
        min_letters: Minimum number of letters required (default: 3)

    Returns:
        Cleaned username or "Anonymous" if invalid

    Examples:
        >>> clean_username("Hasan855")
        'Hasan'
        >>> clean_username("Mo7amed")
        'Mohamed'
        >>> clean_username("___")
        'Anonymous'
    """
    if name is None:
        return "Anonymous"

    s = str(name)

    # Normalize separators to spaces
    s = re.sub(r"[_\-./\\|]+", " ", s)

    # Remove punctuation, symbols, emoji
    s = "".join(ch if not _is_punct_or_symbol(ch) else " " for ch in s)
    s = re.sub(r"\s+", " ", s).strip()

    if not s:
        return "Anonymous"

    tokens = []
    for tok in s.split():
        # Remove trailing digits (e.g., Hasan855 -> Hasan)
        if re.search(r"[A-Za-z]", tok) and re.search(r"\d+$", tok):
            tok = re.sub(r"\d+$", "", tok)

        # Arabizi: convert digits inside Latin words
        if re.search(r"[A-Za-z]", tok) and re.search(r"\d", tok):
            for digit, replacement in ARABIZI_MAP.items():
                tok = tok.replace(digit, replacement)
            tok = re.sub(r"\d+", "", tok)

        # Skip pure numbers
        if re.fullmatch(r"\d+", tok):
            continue

        tokens.append(tok)

    result = " ".join(tokens).strip()

    # Count letters only (across all scripts)
    letter_count = sum(
        1 for ch in result if unicodedata.category(ch).startswith("L")
    )

    if letter_count < min_letters:
        return "Anonymous"

    return result


class UsernamePreprocessor:
    """
    Preprocessor for username columns in DataFrames.

    Attributes:
        column_name: Name of the column containing usernames
        output_column: Name for the cleaned username column
        min_letters: Minimum letters required for valid username
    """

    def __init__(
        self,
        column_name: str = "User Name",
        output_column: str = "clean_name",
        min_letters: int = 3
    ):
        self.column_name = column_name
        self.output_column = output_column
        self.min_letters = min_letters

    def fit(self, df: pd.DataFrame) -> "UsernamePreprocessor":
        """Fit method (no-op, for sklearn compatibility)."""
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform DataFrame by cleaning usernames.

        Args:
            df: Input DataFrame with username column

        Returns:
            DataFrame with added clean_name column
        """
        df = df.copy()

        # Fill null usernames
        df[self.column_name] = df[self.column_name].fillna("Anonymous")

        # Apply cleaning
        df[self.output_column] = df[self.column_name].apply(
            lambda x: clean_username(x, self.min_letters)
        )

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)
