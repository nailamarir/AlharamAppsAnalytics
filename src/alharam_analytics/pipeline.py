"""
Main preprocessing pipeline for AlHaram Analytics.

Orchestrates all preprocessing steps in a configurable pipeline.
"""

from typing import Optional, List
from pathlib import Path

import pandas as pd

from .preprocessing import UsernamePreprocessor, LanguageDetector, TextCleaner
from .feature_engineering.period_tagger import PeriodTagger
from .feature_engineering.device_mapper import DeviceTypeMapper
from .feature_engineering.app_name_normalizer import AppNameNormalizer
from .feature_engineering.service_classifier import ServiceClassifier
from .gender_prediction import GenderEnsemblePredictor
from .utils import load_data, save_data


class PreprocessingPipeline:
    """
    Main preprocessing pipeline for app review data.

    Combines all preprocessing steps:
    1. Text cleaning (URL/emoji/hashtag extraction, Arabic normalization)
    2. Username cleaning
    3. Language detection
    4. Device type mapping
    5. App name normalization
    6. Service type classification
    7. Period tagging
    8. Gender prediction (optional)

    Example:
        >>> pipeline = PreprocessingPipeline()
        >>> df = pipeline.run("data/raw/reviews.xlsx")
        >>> pipeline.save(df, "data/processed/reviews_clean.xlsx")
    """

    def __init__(
        self,
        include_gender_prediction: bool = False,
        include_text_cleaning: bool = True,
        verbose: bool = True
    ):
        """
        Initialize the pipeline.

        Args:
            include_gender_prediction: Whether to include gender prediction
                (requires HuggingFace transformers)
            include_text_cleaning: Whether to include text cleaning step
            verbose: Whether to print progress messages
        """
        self.include_gender_prediction = include_gender_prediction
        self.include_text_cleaning = include_text_cleaning
        self.verbose = verbose

        # Initialize processors
        self.text_cleaner = TextCleaner(verbose=verbose) if include_text_cleaning else None
        self.username_preprocessor = UsernamePreprocessor()
        self.language_detector = LanguageDetector()
        self.device_mapper = DeviceTypeMapper()
        self.app_normalizer = AppNameNormalizer()
        self.service_classifier = ServiceClassifier()
        self.period_tagger = PeriodTagger()

        if include_gender_prediction:
            self.gender_predictor = GenderEnsemblePredictor()
        else:
            self.gender_predictor = None

    def _log(self, message: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(f"[Pipeline] {message}")

    def run(
        self,
        data: pd.DataFrame | str | Path,
        steps: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Run the preprocessing pipeline.

        Args:
            data: Input DataFrame or path to data file
            steps: Optional list of steps to run. If None, runs all steps.
                   Available steps: "text_clean", "username", "language",
                   "device", "app_name", "service", "period", "gender"

        Returns:
            Processed DataFrame
        """
        # Load data if path provided
        if isinstance(data, (str, Path)):
            self._log(f"Loading data from {data}")
            df = load_data(data)
        else:
            df = data.copy()

        self._log(f"Starting pipeline with {len(df)} rows")

        # Default to all steps
        if steps is None:
            steps = ["text_clean", "username", "language", "device", "app_name",
                     "service", "period"]
            if self.include_gender_prediction:
                steps.append("gender")

        # Run selected steps
        if "text_clean" in steps and self.text_cleaner:
            self._log("Cleaning text (extracting URLs, emojis, normalizing Arabic)...")
            df = self.text_cleaner.transform(df)

        if "username" in steps:
            self._log("Cleaning usernames...")
            df = self.username_preprocessor.transform(df)

        if "language" in steps:
            self._log("Detecting languages...")
            df = self.language_detector.transform(df)

        if "device" in steps:
            self._log("Mapping device types...")
            df = self.device_mapper.transform(df)

        if "app_name" in steps:
            self._log("Normalizing app names...")
            df = self.app_normalizer.transform(df)

        if "service" in steps:
            self._log("Classifying service types...")
            df = self.service_classifier.transform(df)

        if "period" in steps:
            self._log("Tagging periods...")
            df = self.period_tagger.transform(df)
            df = self.period_tagger.add_quarter_period(df)

        if "gender" in steps and self.gender_predictor:
            self._log("Predicting gender (this may take a while)...")
            df = self.gender_predictor.predict(df)

        self._log(f"Pipeline complete. Output has {len(df.columns)} columns")
        return df

    def save(
        self,
        df: pd.DataFrame,
        output_path: str | Path,
        **kwargs
    ) -> None:
        """
        Save processed DataFrame.

        Args:
            df: Processed DataFrame
            output_path: Output file path
            **kwargs: Additional arguments for save_data
        """
        self._log(f"Saving to {output_path}")
        save_data(df, output_path, **kwargs)


def run_pipeline(
    input_path: str | Path,
    output_path: str | Path,
    include_gender: bool = False
) -> pd.DataFrame:
    """
    Convenience function to run the full pipeline.

    Args:
        input_path: Path to input data file
        output_path: Path to save processed data
        include_gender: Whether to include gender prediction

    Returns:
        Processed DataFrame
    """
    pipeline = PreprocessingPipeline(include_gender_prediction=include_gender)
    df = pipeline.run(input_path)
    pipeline.save(df, output_path)
    return df
