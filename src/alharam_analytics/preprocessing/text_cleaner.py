"""
Text Cleaner for Arabic App Reviews.

This module provides comprehensive text cleaning and normalization for Arabic text
while preserving all information by extracting elements to separate columns.

Key Features:
- URL extraction (not removal)
- Email extraction (not removal)
- Emoji extraction with sentiment preservation
- Arabic diacritics handling (stored separately)
- Arabic character normalization
- Repeated character collapse detection
- Hashtag and mention extraction
- Whitespace normalization

Author: AlHaram Analytics Team
"""

import re
import unicodedata
from typing import Optional, List, Dict, Any, Tuple
import pandas as pd


class TextCleaner:
    """
    Arabic text cleaner that extracts and preserves all information.

    Instead of removing data, this cleaner extracts elements (URLs, emails,
    emojis, etc.) to separate columns while creating a normalized version
    for analysis.

    Attributes:
        text_column: Name of the column containing text to clean
        output_column: Name for the cleaned text column
        normalize_arabic: Whether to normalize Arabic characters
        extract_urls: Whether to extract URLs to separate column
        extract_emails: Whether to extract emails to separate column
        extract_emojis: Whether to extract emojis to separate column
        extract_hashtags: Whether to extract hashtags to separate column
        extract_mentions: Whether to extract @mentions to separate column
        collapse_repeated: Whether to collapse repeated characters
        min_repeated: Minimum repetitions before collapsing (default 3)
        normalize_whitespace: Whether to normalize whitespace

    Example:
        >>> cleaner = TextCleaner(
        ...     normalize_arabic=True,
        ...     extract_urls=True,
        ...     extract_emojis=True
        ... )
        >>> df = cleaner.transform(df)
    """

    # Arabic diacritics (tashkeel/harakat) Unicode range
    ARABIC_DIACRITICS = re.compile(r'[\u064B-\u0652\u0670\u0640]')

    # Arabic letter normalization mappings
    ARABIC_NORMALIZATION = {
        # Alef variants -> Alef
        '\u0622': '\u0627',  # آ -> ا (Alef with Madda)
        '\u0623': '\u0627',  # أ -> ا (Alef with Hamza Above)
        '\u0625': '\u0627',  # إ -> ا (Alef with Hamza Below)
        '\u0671': '\u0627',  # ٱ -> ا (Alef Wasla)
        # Alef Maksura -> Ya
        '\u0649': '\u064A',  # ى -> ي
        # Ta Marbuta -> Ha (optional, configurable)
        # '\u0629': '\u0647',  # ة -> ه
    }

    # URL pattern
    URL_PATTERN = re.compile(
        r'https?://[^\s<>"{}|\\^`\[\]]+|'
        r'www\.[^\s<>"{}|\\^`\[\]]+'
    )

    # Email pattern
    EMAIL_PATTERN = re.compile(
        r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    )

    # Hashtag pattern (supports Arabic and Latin)
    HASHTAG_PATTERN = re.compile(
        r'#[\w\u0600-\u06FF]+'
    )

    # Mention pattern
    MENTION_PATTERN = re.compile(
        r'@[\w\u0600-\u06FF]+'
    )

    # Emoji pattern (comprehensive Unicode emoji ranges)
    EMOJI_PATTERN = re.compile(
        r'['
        r'\U0001F600-\U0001F64F'  # Emoticons
        r'\U0001F300-\U0001F5FF'  # Misc Symbols and Pictographs
        r'\U0001F680-\U0001F6FF'  # Transport and Map
        r'\U0001F700-\U0001F77F'  # Alchemical Symbols
        r'\U0001F780-\U0001F7FF'  # Geometric Shapes Extended
        r'\U0001F800-\U0001F8FF'  # Supplemental Arrows-C
        r'\U0001F900-\U0001F9FF'  # Supplemental Symbols and Pictographs
        r'\U0001FA00-\U0001FA6F'  # Chess Symbols
        r'\U0001FA70-\U0001FAFF'  # Symbols and Pictographs Extended-A
        r'\U00002702-\U000027B0'  # Dingbats
        r'\U0001F1E0-\U0001F1FF'  # Flags
        r'\U00002600-\U000026FF'  # Misc symbols
        r'\U00002300-\U000023FF'  # Misc Technical
        r']+'
    )

    # Repeated character pattern (3+ repetitions)
    REPEATED_CHAR_PATTERN = re.compile(r'(.)\1{2,}')

    # Zero-width characters
    ZERO_WIDTH_CHARS = re.compile(r'[\u200B-\u200D\uFEFF\u00AD]')

    # Multiple whitespace
    MULTI_WHITESPACE = re.compile(r'\s+')

    def __init__(
        self,
        text_column: str = "Review Text",
        output_column: str = "clean_text",
        normalize_arabic: bool = True,
        normalize_ta_marbuta: bool = False,
        extract_urls: bool = True,
        extract_emails: bool = True,
        extract_emojis: bool = True,
        extract_hashtags: bool = True,
        extract_mentions: bool = True,
        collapse_repeated: bool = True,
        min_repeated: int = 3,
        normalize_whitespace: bool = True,
        strip_diacritics: bool = True,
        verbose: bool = False
    ):
        """
        Initialize the TextCleaner.

        Args:
            text_column: Name of the input text column
            output_column: Name for the cleaned text output column
            normalize_arabic: Normalize Arabic character variants
            normalize_ta_marbuta: Convert Ta Marbuta to Ha (ة -> ه)
            extract_urls: Extract URLs to separate column
            extract_emails: Extract emails to separate column
            extract_emojis: Extract emojis to separate column
            extract_hashtags: Extract hashtags to separate column
            extract_mentions: Extract @mentions to separate column
            collapse_repeated: Collapse repeated characters
            min_repeated: Minimum repetitions to trigger collapse
            normalize_whitespace: Normalize whitespace characters
            strip_diacritics: Remove Arabic diacritics from clean text
            verbose: Print processing information
        """
        self.text_column = text_column
        self.output_column = output_column
        self.normalize_arabic = normalize_arabic
        self.normalize_ta_marbuta = normalize_ta_marbuta
        self.extract_urls = extract_urls
        self.extract_emails = extract_emails
        self.extract_emojis = extract_emojis
        self.extract_hashtags = extract_hashtags
        self.extract_mentions = extract_mentions
        self.collapse_repeated = collapse_repeated
        self.min_repeated = min_repeated
        self.normalize_whitespace = normalize_whitespace
        self.strip_diacritics = strip_diacritics
        self.verbose = verbose

        # Build normalization map
        self._build_normalization_map()

    def _build_normalization_map(self) -> None:
        """Build the Arabic character normalization mapping."""
        self.norm_map = dict(self.ARABIC_NORMALIZATION)
        if self.normalize_ta_marbuta:
            self.norm_map['\u0629'] = '\u0647'  # ة -> ه

    def fit(self, X: pd.DataFrame, y=None) -> 'TextCleaner':
        """Fit method for sklearn compatibility (no-op)."""
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the DataFrame by cleaning text and extracting elements.

        Args:
            df: Input DataFrame with text column

        Returns:
            DataFrame with cleaned text and extracted element columns
        """
        df = df.copy()

        if self.text_column not in df.columns:
            if self.verbose:
                print(f"Warning: Column '{self.text_column}' not found. Skipping text cleaning.")
            return df

        if self.verbose:
            print(f"Processing {len(df)} rows...")

        # Initialize new columns
        results = df[self.text_column].apply(self._process_text)

        # Unpack results into DataFrame columns
        df[self.output_column] = results.apply(lambda x: x['clean_text'])

        if self.extract_urls:
            df['extracted_urls'] = results.apply(lambda x: x['urls'])
            df['url_count'] = results.apply(lambda x: len(x['urls']))
            df['has_urls'] = df['url_count'] > 0

        if self.extract_emails:
            df['extracted_emails'] = results.apply(lambda x: x['emails'])
            df['email_count'] = results.apply(lambda x: len(x['emails']))
            df['has_emails'] = df['email_count'] > 0

        if self.extract_emojis:
            df['extracted_emojis'] = results.apply(lambda x: x['emojis'])
            df['emoji_count'] = results.apply(lambda x: len(x['emojis']))
            df['has_emojis'] = df['emoji_count'] > 0

        if self.extract_hashtags:
            df['extracted_hashtags'] = results.apply(lambda x: x['hashtags'])
            df['hashtag_count'] = results.apply(lambda x: len(x['hashtags']))
            df['has_hashtags'] = df['hashtag_count'] > 0

        if self.extract_mentions:
            df['extracted_mentions'] = results.apply(lambda x: x['mentions'])
            df['mention_count'] = results.apply(lambda x: len(x['mentions']))
            df['has_mentions'] = df['mention_count'] > 0

        if self.strip_diacritics:
            df['has_diacritics'] = results.apply(lambda x: x['had_diacritics'])

        if self.collapse_repeated:
            df['has_elongation'] = results.apply(lambda x: x['had_elongation'])

        if self.verbose:
            self._print_summary(df)

        return df

    def fit_transform(self, df: pd.DataFrame, y=None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df, y).transform(df)

    def _process_text(self, text: Any) -> Dict[str, Any]:
        """
        Process a single text value.

        Args:
            text: Input text (can be None or non-string)

        Returns:
            Dictionary with cleaned text and extracted elements
        """
        result = {
            'clean_text': '',
            'urls': [],
            'emails': [],
            'emojis': [],
            'hashtags': [],
            'mentions': [],
            'had_diacritics': False,
            'had_elongation': False
        }

        # Handle null/empty values
        if pd.isna(text) or text is None:
            return result

        text = str(text).strip()
        if not text:
            return result

        clean = text

        # 1. Extract URLs (before other processing)
        if self.extract_urls:
            result['urls'] = self.URL_PATTERN.findall(clean)
            clean = self.URL_PATTERN.sub(' ', clean)

        # 2. Extract emails
        if self.extract_emails:
            result['emails'] = self.EMAIL_PATTERN.findall(clean)
            clean = self.EMAIL_PATTERN.sub(' ', clean)

        # 3. Extract hashtags
        if self.extract_hashtags:
            result['hashtags'] = self.HASHTAG_PATTERN.findall(clean)
            clean = self.HASHTAG_PATTERN.sub(' ', clean)

        # 4. Extract mentions
        if self.extract_mentions:
            result['mentions'] = self.MENTION_PATTERN.findall(clean)
            clean = self.MENTION_PATTERN.sub(' ', clean)

        # 5. Extract emojis
        if self.extract_emojis:
            emojis_found = self.EMOJI_PATTERN.findall(clean)
            # Flatten emoji strings into individual emojis
            result['emojis'] = list(''.join(emojis_found))
            clean = self.EMOJI_PATTERN.sub(' ', clean)

        # 6. Check for and handle diacritics
        if self.strip_diacritics:
            if self.ARABIC_DIACRITICS.search(clean):
                result['had_diacritics'] = True
                clean = self.ARABIC_DIACRITICS.sub('', clean)

        # 7. Arabic character normalization
        if self.normalize_arabic:
            clean = self._normalize_arabic(clean)

        # 8. Collapse repeated characters
        if self.collapse_repeated:
            clean, had_elongation = self._collapse_repeated(clean)
            result['had_elongation'] = had_elongation

        # 9. Remove zero-width characters
        clean = self.ZERO_WIDTH_CHARS.sub('', clean)

        # 10. Normalize whitespace
        if self.normalize_whitespace:
            clean = self.MULTI_WHITESPACE.sub(' ', clean).strip()

        result['clean_text'] = clean
        return result

    def _normalize_arabic(self, text: str) -> str:
        """
        Normalize Arabic character variants.

        Args:
            text: Input text

        Returns:
            Text with normalized Arabic characters
        """
        for original, replacement in self.norm_map.items():
            text = text.replace(original, replacement)
        return text

    def _collapse_repeated(self, text: str) -> Tuple[str, bool]:
        """
        Collapse repeated characters while preserving meaning.

        Args:
            text: Input text

        Returns:
            Tuple of (collapsed text, whether elongation was found)
        """
        pattern = re.compile(r'(.)\1{' + str(self.min_repeated - 1) + r',}')
        had_elongation = bool(pattern.search(text))

        # Collapse to 2 characters (preserves some emphasis)
        collapsed = pattern.sub(r'\1\1', text)

        return collapsed, had_elongation

    def _print_summary(self, df: pd.DataFrame) -> None:
        """Print processing summary statistics."""
        print("\n" + "=" * 50)
        print("Text Cleaning Summary")
        print("=" * 50)
        print(f"Total rows processed: {len(df)}")

        if self.extract_urls and 'has_urls' in df.columns:
            print(f"Rows with URLs: {df['has_urls'].sum()} ({100*df['has_urls'].mean():.1f}%)")

        if self.extract_emails and 'has_emails' in df.columns:
            print(f"Rows with emails: {df['has_emails'].sum()} ({100*df['has_emails'].mean():.1f}%)")

        if self.extract_emojis and 'has_emojis' in df.columns:
            print(f"Rows with emojis: {df['has_emojis'].sum()} ({100*df['has_emojis'].mean():.1f}%)")
            print(f"Total emojis extracted: {df['emoji_count'].sum()}")

        if self.extract_hashtags and 'has_hashtags' in df.columns:
            print(f"Rows with hashtags: {df['has_hashtags'].sum()} ({100*df['has_hashtags'].mean():.1f}%)")

        if self.extract_mentions and 'has_mentions' in df.columns:
            print(f"Rows with mentions: {df['has_mentions'].sum()} ({100*df['has_mentions'].mean():.1f}%)")

        if self.strip_diacritics and 'has_diacritics' in df.columns:
            print(f"Rows with diacritics: {df['has_diacritics'].sum()} ({100*df['has_diacritics'].mean():.1f}%)")

        if self.collapse_repeated and 'has_elongation' in df.columns:
            print(f"Rows with elongation: {df['has_elongation'].sum()} ({100*df['has_elongation'].mean():.1f}%)")

        print("=" * 50)

    def get_emoji_sentiment_hints(self) -> Dict[str, str]:
        """
        Get common emoji sentiment mappings.

        Returns:
            Dictionary mapping emojis to sentiment hints
        """
        return {
            # Positive
            '😀': 'positive', '😃': 'positive', '😄': 'positive',
            '😊': 'positive', '😍': 'positive', '🥰': 'positive',
            '👍': 'positive', '❤️': 'positive', '💯': 'positive',
            '✨': 'positive', '🎉': 'positive', '👏': 'positive',
            '💪': 'positive', '🙏': 'positive', '😇': 'positive',

            # Negative
            '😢': 'negative', '😭': 'negative', '😡': 'negative',
            '😠': 'negative', '👎': 'negative', '💔': 'negative',
            '😤': 'negative', '😞': 'negative', '😔': 'negative',
            '🤬': 'negative', '😒': 'negative', '🙄': 'negative',

            # Neutral/Mixed
            '🤔': 'neutral', '😐': 'neutral', '😶': 'neutral',
            '🤷': 'neutral', '😅': 'mixed', '😬': 'mixed'
        }


# Convenience function for quick usage
def clean_arabic_text(
    df: pd.DataFrame,
    text_column: str = "Review Text",
    **kwargs
) -> pd.DataFrame:
    """
    Quick function to clean Arabic text in a DataFrame.

    Args:
        df: Input DataFrame
        text_column: Name of the text column
        **kwargs: Additional arguments for TextCleaner

    Returns:
        DataFrame with cleaned text and extracted columns
    """
    cleaner = TextCleaner(text_column=text_column, **kwargs)
    return cleaner.transform(df)
