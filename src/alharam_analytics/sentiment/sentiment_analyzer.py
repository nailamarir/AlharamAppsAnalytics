"""
Deep Learning Sentiment Analyzer for Arabic App Reviews.

Uses pre-trained transformer models (AraBERT/CAMeL-BERT) for
sentiment classification of Arabic and mixed-language text.

Author: AlHaram Analytics Team
"""

import warnings
from typing import Optional, List, Dict, Any, Tuple
import pandas as pd
import numpy as np

# Suppress warnings during import
warnings.filterwarnings('ignore')

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class SentimentAnalyzer:
    """
    Deep learning-based sentiment analyzer for Arabic app reviews.

    Uses pre-trained Arabic BERT models for sentiment classification.
    Supports multiple model options:
    - CAMeL-BERT (recommended for MSA/Dialectal Arabic)
    - AraBERT (good for MSA)
    - Multilingual BERT (fallback for mixed content)

    Attributes:
        model_name: HuggingFace model identifier
        batch_size: Batch size for inference
        max_length: Maximum token length
        device: Computing device (cuda/cpu)

    Example:
        >>> analyzer = SentimentAnalyzer()
        >>> df = analyzer.transform(df)
        >>> # Adds: sentiment, sentiment_score, sentiment_confidence
    """

    # Available pre-trained models for Arabic sentiment
    AVAILABLE_MODELS = {
        'camel-bert': 'CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment',
        'arabert': 'aubmindlab/bert-base-arabertv2',
        'multilingual': 'nlptown/bert-base-multilingual-uncased-sentiment',
    }

    # Sentiment label mappings
    SENTIMENT_LABELS = {
        0: 'negative',
        1: 'neutral',
        2: 'positive'
    }

    # For 5-class models (like multilingual)
    SENTIMENT_LABELS_5 = {
        0: 'very_negative',
        1: 'negative',
        2: 'neutral',
        3: 'positive',
        4: 'very_positive'
    }

    def __init__(
        self,
        model_name: str = 'camel-bert',
        text_column: str = "Review Text",
        batch_size: int = 16,
        max_length: int = 128,
        device: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Initialize the SentimentAnalyzer.

        Args:
            model_name: Model to use ('camel-bert', 'arabert', 'multilingual')
                       or a HuggingFace model path
            text_column: Name of the column containing text
            batch_size: Batch size for inference
            max_length: Maximum token sequence length
            device: Device to use ('cuda', 'cpu', or None for auto)
            verbose: Print progress messages
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers and torch are required for sentiment analysis. "
                "Install with: pip install transformers torch"
            )

        self.text_column = text_column
        self.batch_size = batch_size
        self.max_length = max_length
        self.verbose = verbose

        # Resolve model name
        if model_name in self.AVAILABLE_MODELS:
            self.model_path = self.AVAILABLE_MODELS[model_name]
        else:
            self.model_path = model_name

        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        if self.verbose:
            print(f"Loading sentiment model: {self.model_path}")
            print(f"Device: {self.device}")

        # Load model and tokenizer
        self._load_model()

    def _load_model(self) -> None:
        """Load the pre-trained model and tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path
            )
            self.model.to(self.device)
            self.model.eval()

            # Determine number of labels
            self.num_labels = self.model.config.num_labels

            if self.verbose:
                print(f"Model loaded successfully ({self.num_labels} classes)")

        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def fit(self, X: pd.DataFrame, y=None) -> 'SentimentAnalyzer':
        """Fit method for sklearn compatibility (no-op)."""
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze sentiment for all texts in DataFrame.

        Args:
            df: Input DataFrame with text column

        Returns:
            DataFrame with sentiment columns added:
            - sentiment: Predicted sentiment label
            - sentiment_score: Numeric score (-1 to 1)
            - sentiment_confidence: Model confidence (0 to 1)
        """
        df = df.copy()

        if self.text_column not in df.columns:
            if self.verbose:
                print(f"Warning: Column '{self.text_column}' not found.")
            return df

        if self.verbose:
            print(f"Analyzing sentiment for {len(df)} reviews...")

        # Get texts
        texts = df[self.text_column].fillna('').astype(str).tolist()

        # Process in batches
        all_results = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_results = self._predict_batch(batch_texts)
            all_results.extend(batch_results)

            if self.verbose and (i // self.batch_size + 1) % 10 == 0:
                print(f"  Processed {i + len(batch_texts)}/{len(texts)} reviews")

        # Add results to DataFrame
        df['sentiment'] = [r['label'] for r in all_results]
        df['sentiment_score'] = [r['score'] for r in all_results]
        df['sentiment_confidence'] = [r['confidence'] for r in all_results]

        if self.verbose:
            self._print_summary(df)

        return df

    def fit_transform(self, df: pd.DataFrame, y=None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df, y).transform(df)

    def _predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Predict sentiment for a batch of texts.

        Args:
            texts: List of text strings

        Returns:
            List of result dictionaries
        """
        results = []

        # Handle empty texts
        processed_texts = [t if t.strip() else '[UNK]' for t in texts]

        try:
            # Tokenize
            encodings = self.tokenizer(
                processed_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )

            # Move to device
            encodings = {k: v.to(self.device) for k, v in encodings.items()}

            # Inference
            with torch.no_grad():
                outputs = self.model(**encodings)
                probs = torch.softmax(outputs.logits, dim=-1)
                predictions = torch.argmax(probs, dim=-1)
                confidences = torch.max(probs, dim=-1).values

            # Convert to results
            predictions = predictions.cpu().numpy()
            confidences = confidences.cpu().numpy()
            probs = probs.cpu().numpy()

            for idx, (pred, conf, prob) in enumerate(zip(predictions, confidences, probs)):
                # Map to label
                if self.num_labels == 3:
                    label = self.SENTIMENT_LABELS.get(pred, 'unknown')
                    # Score: -1 (negative) to 1 (positive)
                    score = prob[2] - prob[0]  # positive - negative
                elif self.num_labels == 5:
                    label = self.SENTIMENT_LABELS_5.get(pred, 'unknown')
                    # Map to simpler label
                    if pred <= 1:
                        simple_label = 'negative'
                    elif pred == 2:
                        simple_label = 'neutral'
                    else:
                        simple_label = 'positive'
                    label = simple_label
                    # Score based on weighted sum
                    score = (prob[3] + prob[4]) - (prob[0] + prob[1])
                else:
                    # Binary or other
                    label = 'positive' if pred == 1 else 'negative'
                    score = prob[1] - prob[0] if len(prob) > 1 else 0

                results.append({
                    'label': label,
                    'score': round(float(score), 4),
                    'confidence': round(float(conf), 4)
                })

        except Exception as e:
            # Fallback for errors
            for _ in texts:
                results.append({
                    'label': 'unknown',
                    'score': 0.0,
                    'confidence': 0.0
                })
            if self.verbose:
                print(f"Warning: Batch prediction error: {e}")

        return results

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict sentiment for a single text.

        Args:
            text: Input text string

        Returns:
            Dictionary with label, score, and confidence
        """
        results = self._predict_batch([text])
        return results[0]

    def _print_summary(self, df: pd.DataFrame) -> None:
        """Print analysis summary."""
        print("\n" + "=" * 50)
        print("Sentiment Analysis Summary")
        print("=" * 50)
        print(f"Total reviews analyzed: {len(df)}")

        if 'sentiment' in df.columns:
            sentiment_counts = df['sentiment'].value_counts()
            for label, count in sentiment_counts.items():
                pct = 100 * count / len(df)
                print(f"  {label}: {count} ({pct:.1f}%)")

            avg_score = df['sentiment_score'].mean()
            avg_conf = df['sentiment_confidence'].mean()
            print(f"\nAverage sentiment score: {avg_score:.3f}")
            print(f"Average confidence: {avg_conf:.3f}")

        print("=" * 50)


class SimpleSentimentAnalyzer:
    """
    Lightweight rule-based sentiment analyzer as fallback.

    Uses lexicon-based approach for when deep learning models
    are not available or too slow.

    Suitable for quick analysis or systems without GPU.
    """

    # Arabic positive words
    POSITIVE_WORDS = {
        'ممتاز', 'رائع', 'جميل', 'مذهل', 'شكرا', 'شكراً', 'احسنتم',
        'مفيد', 'سهل', 'سريع', 'جيد', 'حلو', 'تمام', 'افضل', 'أفضل',
        'excellent', 'great', 'good', 'amazing', 'wonderful', 'perfect',
        'love', 'best', 'helpful', 'easy', 'fast', 'thanks', 'thank',
        '👍', '❤️', '😍', '🎉', '✨', '💯', '👏', '🙏', '😊', '😃'
    }

    # Arabic negative words
    NEGATIVE_WORDS = {
        'سيء', 'سيئ', 'فاشل', 'مشكلة', 'مشاكل', 'بطيء', 'صعب',
        'معطل', 'خطأ', 'اخطاء', 'أخطاء', 'زفت', 'قرف', 'مقرف',
        'bad', 'terrible', 'awful', 'horrible', 'worst', 'hate',
        'problem', 'issue', 'bug', 'crash', 'slow', 'difficult',
        '👎', '😡', '😠', '💔', '😢', '😭', '🤬', '😤'
    }

    def __init__(
        self,
        text_column: str = "Review Text",
        verbose: bool = False
    ):
        """
        Initialize SimpleSentimentAnalyzer.

        Args:
            text_column: Name of text column
            verbose: Print progress messages
        """
        self.text_column = text_column
        self.verbose = verbose

    def fit(self, X: pd.DataFrame, y=None) -> 'SimpleSentimentAnalyzer':
        """Fit method for sklearn compatibility (no-op)."""
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze sentiment using lexicon-based approach.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with sentiment columns
        """
        df = df.copy()

        if self.text_column not in df.columns:
            return df

        results = df[self.text_column].apply(self._analyze_text)

        df['sentiment'] = results.apply(lambda x: x['label'])
        df['sentiment_score'] = results.apply(lambda x: x['score'])
        df['sentiment_confidence'] = results.apply(lambda x: x['confidence'])

        return df

    def _analyze_text(self, text: Any) -> Dict[str, Any]:
        """Analyze a single text."""
        if pd.isna(text) or not text:
            return {'label': 'neutral', 'score': 0.0, 'confidence': 0.0}

        text = str(text).lower()
        words = set(text.split())

        pos_count = len(words & self.POSITIVE_WORDS)
        neg_count = len(words & self.NEGATIVE_WORDS)

        # Check for emoji matches
        for char in text:
            if char in self.POSITIVE_WORDS:
                pos_count += 1
            if char in self.NEGATIVE_WORDS:
                neg_count += 1

        total = pos_count + neg_count
        if total == 0:
            return {'label': 'neutral', 'score': 0.0, 'confidence': 0.3}

        score = (pos_count - neg_count) / total
        confidence = min(total / 5, 1.0)  # More words = higher confidence

        if score > 0.2:
            label = 'positive'
        elif score < -0.2:
            label = 'negative'
        else:
            label = 'neutral'

        return {
            'label': label,
            'score': round(score, 4),
            'confidence': round(confidence, 4)
        }


def analyze_sentiment(
    df: pd.DataFrame,
    text_column: str = "Review Text",
    use_deep_learning: bool = True,
    **kwargs
) -> pd.DataFrame:
    """
    Quick function to analyze sentiment in a DataFrame.

    Args:
        df: Input DataFrame
        text_column: Name of text column
        use_deep_learning: Use transformer model (True) or lexicon (False)
        **kwargs: Additional arguments for analyzer

    Returns:
        DataFrame with sentiment columns
    """
    if use_deep_learning and TRANSFORMERS_AVAILABLE:
        analyzer = SentimentAnalyzer(text_column=text_column, **kwargs)
    else:
        analyzer = SimpleSentimentAnalyzer(text_column=text_column, **kwargs)

    return analyzer.transform(df)
