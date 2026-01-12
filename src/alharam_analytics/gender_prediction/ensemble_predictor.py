"""
Ensemble gender prediction using multiple models.

Combines predictions from multiple gender classifiers for more accurate results.
"""

from typing import List, Optional
import numpy as np
import pandas as pd

from .hf_gender_classifier import (
    HFGenderClassifier,
    create_genderize_classifier,
    create_gender_classification_classifier
)


class GenderEnsemblePredictor:
    """
    Ensemble predictor that combines multiple gender classifiers.

    Uses agreement between models and confidence thresholds to determine
    the final gender prediction.

    Attributes:
        classifiers: List of HFGenderClassifier instances
        high_confidence_threshold: Threshold for trusting single model (default: 0.80)
    """

    def __init__(
        self,
        classifiers: Optional[List[HFGenderClassifier]] = None,
        high_confidence_threshold: float = 0.80
    ):
        self.classifiers = classifiers
        self.high_confidence_threshold = high_confidence_threshold
        self._initialized = False

    def _init_classifiers(self):
        """Initialize default classifiers if not provided."""
        if not self._initialized:
            if self.classifiers is None:
                self.classifiers = [
                    create_genderize_classifier(),
                    create_gender_classification_classifier()
                ]
            self._initialized = True

    def predict_one(self, name: str) -> dict:
        """
        Predict gender for a single name using all classifiers.

        Args:
            name: Username or name string

        Returns:
            Dictionary with predictions from all models and final result
        """
        self._init_classifiers()

        if pd.isna(name) or not str(name).strip():
            return {
                "pred_gender_1": "unknown",
                "pred_score_1": np.nan,
                "pred_gender_2": "unknown",
                "pred_score_2": np.nan,
                "gender_final": "unknown"
            }

        results = []
        for clf in self.classifiers:
            label, score = clf.predict_one(name)
            results.append((label, score))

        # Combine results
        g1, s1 = results[0] if results else ("unknown", np.nan)
        g2, s2 = results[1] if len(results) > 1 else ("unknown", np.nan)

        final_gender = self._determine_final_gender(g1, s1, g2, s2)

        return {
            "pred_gender_1": g1,
            "pred_score_1": s1,
            "pred_gender_2": g2,
            "pred_score_2": s2,
            "gender_final": final_gender
        }

    def _determine_final_gender(
        self, g1: str, s1: float, g2: str, s2: float
    ) -> str:
        """
        Determine final gender based on model agreement and confidence.

        Logic:
        1. If both models agree and neither is unknown/error -> use that gender
        2. If one model has high confidence (>=0.80) -> trust that model
        3. Otherwise -> unknown
        """
        invalid_labels = {"unknown", "Error", None}

        # Case 1: Models agree
        if g1 == g2 and g1 not in invalid_labels:
            return g1

        # Case 2: First model has high confidence
        if not np.isnan(s1) and s1 >= self.high_confidence_threshold:
            if g1 not in invalid_labels:
                return g1

        # Case 3: Second model has high confidence
        if not np.isnan(s2) and s2 >= self.high_confidence_threshold:
            if g2 not in invalid_labels:
                return g2

        return "unknown"

    def predict(
        self,
        df: pd.DataFrame,
        name_column: str = "clean_name"
    ) -> pd.DataFrame:
        """
        Predict gender for all names in a DataFrame.

        Args:
            df: Input DataFrame
            name_column: Column containing names to classify

        Returns:
            DataFrame with gender predictions added
        """
        df = df.copy()

        predictions = df[name_column].apply(self.predict_one)

        df["pred_gender_1"] = predictions.apply(lambda x: x["pred_gender_1"])
        df["pred_score_1"] = predictions.apply(lambda x: x["pred_score_1"])
        df["pred_gender_2"] = predictions.apply(lambda x: x["pred_gender_2"])
        df["pred_score_2"] = predictions.apply(lambda x: x["pred_score_2"])
        df["gender_final"] = predictions.apply(lambda x: x["gender_final"])

        return df
