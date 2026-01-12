"""
Language detection for review text.

Detects Arabic, English, Mixed, and Unknown languages.
"""

import re
from typing import Optional

import pandas as pd

try:
    import langid
    LANGID_AVAILABLE = True
except ImportError:
    LANGID_AVAILABLE = False


def detect_language(text: Optional[str]) -> str:
    """
    Detect the language of a text string.

    Args:
        text: Input text to analyze

    Returns:
        One of: "Arabic", "English", "Mixed", "Unknown"

    Examples:
        >>> detect_language("Hello world")
        'English'
        >>> detect_language("مرحبا")
        'Arabic'
        >>> detect_language("Hello مرحبا")
        'Mixed'
    """
    if not LANGID_AVAILABLE:
        raise ImportError("langid package is required. Install with: pip install langid")

    text = str(text).strip() if text else ""

    if not text:
        return "Unknown"

    # Detect main language
    lang = langid.classify(text)[0]

    # Check for Arabic and Latin scripts
    has_arabic = bool(re.search(r"[\u0600-\u06FF]", text))
    has_english = bool(re.search(r"[A-Za-z]", text))

    if has_arabic and has_english:
        return "Mixed"
    elif lang == "ar" or has_arabic:
        return "Arabic"
    elif lang == "en" or has_english:
        return "English"
    else:
        return "Unknown"


class LanguageDetector:
    """
    Language detector for DataFrame text columns.

    Attributes:
        column_name: Name of the column to analyze
        output_column: Name for the language column
    """

    def __init__(
        self,
        column_name: str = "Review Text",
        output_column: str = "language"
    ):
        self.column_name = column_name
        self.output_column = output_column

    def fit(self, df: pd.DataFrame) -> "LanguageDetector":
        """Fit method (no-op, for sklearn compatibility)."""
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform DataFrame by detecting language.

        Args:
            df: Input DataFrame with text column

        Returns:
            DataFrame with added language column
        """
        df = df.copy()
        df[self.output_column] = df[self.column_name].apply(detect_language)
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)
