"""
HuggingFace-based gender classification from names.

Uses transformer models to predict gender from usernames.
"""

from typing import Optional, Tuple
import numpy as np
import pandas as pd


class HFGenderClassifier:
    """
    Gender classifier using HuggingFace transformers.

    Attributes:
        model_name: HuggingFace model identifier
        confidence_threshold: Minimum confidence for non-unknown predictions
        label_map: Mapping from model labels to gender strings
    """

    def __init__(
        self,
        model_name: str = "padmajabfrl/Gender-Classification",
        confidence_threshold: float = 0.60,
        label_map: Optional[dict] = None
    ):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.label_map = label_map or {}
        self._pipeline = None

    def _load_pipeline(self):
        """Lazy load the HuggingFace pipeline."""
        if self._pipeline is None:
            try:
                from transformers import pipeline
                self._pipeline = pipeline(
                    task="text-classification",
                    model=self.model_name,
                    tokenizer=self.model_name
                )
            except ImportError:
                raise ImportError(
                    "transformers and torch packages required. "
                    "Install with: pip install transformers torch"
                )
        return self._pipeline

    def predict_one(self, name: str) -> Tuple[str, float]:
        """
        Predict gender for a single name.

        Args:
            name: Username or name string

        Returns:
            Tuple of (gender_label, confidence_score)
        """
        if pd.isna(name) or not str(name).strip():
            return ("unknown", np.nan)

        text = str(name).strip()

        try:
            pipe = self._load_pipeline()
            result = pipe(text, truncation=True)[0]
            label = result["label"]
            score = float(result["score"])

            # Apply label mapping if provided
            if label in self.label_map:
                label = self.label_map[label]

            # Apply confidence threshold
            if score < self.confidence_threshold:
                return ("unknown", score)

            return (label, score)

        except Exception:
            return ("Error", np.nan)

    def predict(self, df: pd.DataFrame, name_column: str = "clean_name") -> pd.DataFrame:
        """
        Predict gender for all names in a DataFrame.

        Args:
            df: Input DataFrame
            name_column: Column containing names to classify

        Returns:
            DataFrame with added pred_gender and pred_gender_score columns
        """
        df = df.copy()

        predictions = df[name_column].apply(self.predict_one)
        df["pred_gender"] = predictions.apply(lambda x: x[0])
        df["pred_gender_score"] = predictions.apply(lambda x: x[1])

        return df


# Pre-configured classifiers for the two models used in the notebooks
def create_genderize_classifier() -> HFGenderClassifier:
    """Create classifier using imranali291/genderize model."""
    return HFGenderClassifier(
        model_name="imranali291/genderize",
        label_map={"LABEL_0": "Female", "LABEL_1": "Male"}
    )


def create_gender_classification_classifier() -> HFGenderClassifier:
    """Create classifier using padmajabfrl/Gender-Classification model."""
    return HFGenderClassifier(
        model_name="padmajabfrl/Gender-Classification"
    )
