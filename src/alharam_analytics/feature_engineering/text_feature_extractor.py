"""
Text Feature Extractor for Arabic App Reviews.

Extracts quantitative features from text for analysis and modeling:
- Length metrics (characters, words, sentences)
- Arabic/English ratio
- Vocabulary richness
- Punctuation and special character usage

Author: AlHaram Analytics Team
"""

import re
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np


class TextFeatureExtractor:
    """
    Extract quantitative text features from Arabic app reviews.

    Features extracted:
    - char_count: Total character count
    - word_count: Total word count
    - sentence_count: Approximate sentence count
    - avg_word_length: Average word length
    - arabic_char_count: Count of Arabic characters
    - latin_char_count: Count of Latin characters
    - arabic_ratio: Ratio of Arabic to total alphabetic chars
    - digit_count: Count of digits
    - punctuation_count: Count of punctuation marks
    - unique_word_count: Number of unique words
    - lexical_diversity: Unique words / total words (TTR)
    - exclamation_count: Count of exclamation marks
    - question_count: Count of question marks

    Example:
        >>> extractor = TextFeatureExtractor()
        >>> df = extractor.transform(df)
    """

    # Arabic character range
    ARABIC_CHARS = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]')

    # Latin character range
    LATIN_CHARS = re.compile(r'[a-zA-Z]')

    # Word pattern (Arabic and Latin)
    WORD_PATTERN = re.compile(r'[\w\u0600-\u06FF]+')

    # Sentence endings
    SENTENCE_ENDINGS = re.compile(r'[.!?؟।।]+')

    # Punctuation
    PUNCTUATION = re.compile(r'[^\w\s\u0600-\u06FF]')

    def __init__(
        self,
        text_column: str = "Review Text",
        prefix: str = "text_",
        include_ratios: bool = True,
        include_lexical: bool = True,
        include_punctuation: bool = True,
        verbose: bool = False
    ):
        """
        Initialize the TextFeatureExtractor.

        Args:
            text_column: Name of the column containing text
            prefix: Prefix for new feature columns
            include_ratios: Include ratio features (arabic_ratio, etc.)
            include_lexical: Include lexical diversity features
            include_punctuation: Include punctuation-related features
            verbose: Print processing information
        """
        self.text_column = text_column
        self.prefix = prefix
        self.include_ratios = include_ratios
        self.include_lexical = include_lexical
        self.include_punctuation = include_punctuation
        self.verbose = verbose

    def fit(self, X: pd.DataFrame, y=None) -> 'TextFeatureExtractor':
        """Fit method for sklearn compatibility (no-op)."""
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract text features from DataFrame.

        Args:
            df: Input DataFrame with text column

        Returns:
            DataFrame with new feature columns
        """
        df = df.copy()

        if self.text_column not in df.columns:
            if self.verbose:
                print(f"Warning: Column '{self.text_column}' not found. Skipping feature extraction.")
            return df

        if self.verbose:
            print(f"Extracting text features from {len(df)} rows...")

        # Extract features for each row
        features = df[self.text_column].apply(self._extract_features)

        # Convert to DataFrame and add prefix
        feature_df = pd.DataFrame(features.tolist())
        feature_df.columns = [f"{self.prefix}{col}" for col in feature_df.columns]

        # Merge with original DataFrame
        for col in feature_df.columns:
            df[col] = feature_df[col].values

        if self.verbose:
            self._print_summary(df, feature_df.columns)

        return df

    def fit_transform(self, df: pd.DataFrame, y=None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df, y).transform(df)

    def _extract_features(self, text: Any) -> Dict[str, Any]:
        """
        Extract all features from a single text value.

        Args:
            text: Input text (can be None or non-string)

        Returns:
            Dictionary of extracted features
        """
        features = {
            'char_count': 0,
            'word_count': 0,
            'sentence_count': 0,
            'avg_word_length': 0.0,
            'arabic_char_count': 0,
            'latin_char_count': 0,
            'digit_count': 0,
        }

        if self.include_ratios:
            features['arabic_ratio'] = 0.0
            features['digit_ratio'] = 0.0

        if self.include_lexical:
            features['unique_word_count'] = 0
            features['lexical_diversity'] = 0.0

        if self.include_punctuation:
            features['punctuation_count'] = 0
            features['exclamation_count'] = 0
            features['question_count'] = 0

        # Handle null/empty values
        if pd.isna(text) or text is None:
            return features

        text = str(text).strip()
        if not text:
            return features

        # Basic counts
        features['char_count'] = len(text)

        # Word extraction and count
        words = self.WORD_PATTERN.findall(text)
        features['word_count'] = len(words)

        # Sentence count
        sentences = self.SENTENCE_ENDINGS.split(text)
        features['sentence_count'] = max(1, len([s for s in sentences if s.strip()]))

        # Average word length
        if words:
            features['avg_word_length'] = round(
                sum(len(w) for w in words) / len(words), 2
            )

        # Arabic character count
        arabic_chars = self.ARABIC_CHARS.findall(text)
        features['arabic_char_count'] = len(arabic_chars)

        # Latin character count
        latin_chars = self.LATIN_CHARS.findall(text)
        features['latin_char_count'] = len(latin_chars)

        # Digit count
        features['digit_count'] = sum(1 for c in text if c.isdigit())

        # Ratio features
        if self.include_ratios:
            total_alpha = features['arabic_char_count'] + features['latin_char_count']
            if total_alpha > 0:
                features['arabic_ratio'] = round(
                    features['arabic_char_count'] / total_alpha, 4
                )

            if features['char_count'] > 0:
                features['digit_ratio'] = round(
                    features['digit_count'] / features['char_count'], 4
                )

        # Lexical diversity features
        if self.include_lexical and words:
            unique_words = set(w.lower() for w in words)
            features['unique_word_count'] = len(unique_words)
            features['lexical_diversity'] = round(
                len(unique_words) / len(words), 4
            )

        # Punctuation features
        if self.include_punctuation:
            features['punctuation_count'] = len(self.PUNCTUATION.findall(text))
            features['exclamation_count'] = text.count('!') + text.count('!')
            features['question_count'] = text.count('?') + text.count('؟')

        return features

    def _print_summary(self, df: pd.DataFrame, feature_cols: list) -> None:
        """Print extraction summary statistics."""
        print("\n" + "=" * 50)
        print("Text Feature Extraction Summary")
        print("=" * 50)
        print(f"Total rows processed: {len(df)}")
        print(f"Features extracted: {len(feature_cols)}")

        # Summary statistics for key features
        prefix = self.prefix
        if f'{prefix}word_count' in df.columns:
            avg_words = df[f'{prefix}word_count'].mean()
            print(f"Average word count: {avg_words:.1f}")

        if f'{prefix}arabic_ratio' in df.columns:
            avg_ratio = df[f'{prefix}arabic_ratio'].mean()
            print(f"Average Arabic ratio: {avg_ratio:.2%}")

        if f'{prefix}lexical_diversity' in df.columns:
            avg_diversity = df[f'{prefix}lexical_diversity'].mean()
            print(f"Average lexical diversity: {avg_diversity:.2f}")

        print("=" * 50)

    def get_feature_names(self) -> list:
        """
        Get list of feature names that will be generated.

        Returns:
            List of feature column names
        """
        base_features = [
            'char_count', 'word_count', 'sentence_count',
            'avg_word_length', 'arabic_char_count', 'latin_char_count',
            'digit_count'
        ]

        if self.include_ratios:
            base_features.extend(['arabic_ratio', 'digit_ratio'])

        if self.include_lexical:
            base_features.extend(['unique_word_count', 'lexical_diversity'])

        if self.include_punctuation:
            base_features.extend(['punctuation_count', 'exclamation_count', 'question_count'])

        return [f"{self.prefix}{f}" for f in base_features]


def extract_text_features(
    df: pd.DataFrame,
    text_column: str = "Review Text",
    **kwargs
) -> pd.DataFrame:
    """
    Quick function to extract text features from a DataFrame.

    Args:
        df: Input DataFrame
        text_column: Name of the text column
        **kwargs: Additional arguments for TextFeatureExtractor

    Returns:
        DataFrame with extracted text features
    """
    extractor = TextFeatureExtractor(text_column=text_column, **kwargs)
    return extractor.transform(df)
