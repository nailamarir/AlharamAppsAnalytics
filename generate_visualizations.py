#!/usr/bin/env python3
"""
Generate visualizations from the AlHaram Analytics preprocessing pipeline.

This script:
1. Loads raw dataset
2. Applies all preprocessing steps
3. Generates evaluation metrics
4. Creates visualization charts
5. Outputs a summary report

Usage:
    python generate_visualizations.py
"""

import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd

# Import pipeline components
from alharam_analytics.preprocessing import UsernamePreprocessor, LanguageDetector, TextCleaner
from alharam_analytics.feature_engineering.period_tagger import PeriodTagger
from alharam_analytics.feature_engineering.device_mapper import DeviceTypeMapper
from alharam_analytics.feature_engineering.app_name_normalizer import AppNameNormalizer
from alharam_analytics.feature_engineering.service_classifier import ServiceClassifier
from alharam_analytics.feature_engineering.text_feature_extractor import TextFeatureExtractor
from alharam_analytics.sentiment import SimpleSentimentAnalyzer, TRANSFORMERS_AVAILABLE
from alharam_analytics.analytics import DatasetAnalyzer, DatasetVisualizer, MATPLOTLIB_AVAILABLE

# Try to import deep learning sentiment if available
if TRANSFORMERS_AVAILABLE:
    from alharam_analytics.sentiment import SentimentAnalyzer


def main():
    print("=" * 60)
    print("ALHARAM ANALYTICS - VISUALIZATION GENERATOR")
    print("=" * 60)
    print()

    # Load dataset
    data_file = PROJECT_ROOT / "webapp" / "uploads" / "Unified_AppReviews_2.xlsx"
    if not data_file.exists():
        data_file = PROJECT_ROOT / "dataset.xlsx"
    if not data_file.exists():
        data_file = PROJECT_ROOT / "data" / "raw" / "dataset.xlsx"

    print(f"Loading dataset from: {data_file}")
    df = pd.read_excel(data_file)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    print()

    # Apply preprocessing pipeline
    print("APPLYING PREPROCESSING PIPELINE")
    print("-" * 40)

    # Step 1: Text Cleaning
    print("Step 1: Text Cleaning...")
    cleaner = TextCleaner(verbose=False)
    df = cleaner.transform(df)
    print(f"  - Added clean_text and extraction columns")

    # Step 2: Username Cleaning
    print("Step 2: Username Cleaning...")
    processor = UsernamePreprocessor()
    df = processor.transform(df)
    print(f"  - Added clean_name column")

    # Step 3: Language Detection
    print("Step 3: Language Detection...")
    detector = LanguageDetector()
    df = detector.transform(df)
    print(f"  - Added language column")

    # Step 4: Device Mapping
    print("Step 4: Device Mapping...")
    mapper = DeviceTypeMapper()
    df = mapper.transform(df)
    print(f"  - Added Device Type column")

    # Step 5: App Name Normalization
    print("Step 5: App Name Normalization...")
    normalizer = AppNameNormalizer()
    df = normalizer.transform(df)
    print(f"  - Normalized Application Name")

    # Step 6: Service Classification
    print("Step 6: Service Classification...")
    classifier = ServiceClassifier()
    df = classifier.transform(df)
    print(f"  - Added Service_Type column")

    # Step 7: Text Feature Extraction
    print("Step 7: Text Feature Extraction...")
    extractor = TextFeatureExtractor(verbose=False)
    df = extractor.transform(df)
    print(f"  - Added 14 text feature columns")

    # Step 8: Period Tagging
    print("Step 8: Period Tagging...")
    tagger = PeriodTagger()
    df = tagger.transform(df)
    df = tagger.add_quarter_period(df)
    print(f"  - Added period and App_Version_Period columns")

    # Step 9: Sentiment Analysis
    print("Step 9: Sentiment Analysis...")
    if TRANSFORMERS_AVAILABLE:
        try:
            analyzer = SentimentAnalyzer(verbose=False)
            df = analyzer.transform(df)
            print(f"  - Used deep learning (CAMeL-BERT)")
        except Exception as e:
            print(f"  - Deep learning failed: {e}")
            analyzer = SimpleSentimentAnalyzer(verbose=False)
            df = analyzer.transform(df)
            print(f"  - Used lexicon-based fallback")
    else:
        analyzer = SimpleSentimentAnalyzer(verbose=False)
        df = analyzer.transform(df)
        print(f"  - Used lexicon-based analyzer (transformers not available)")
    print(f"  - Added sentiment, sentiment_score, sentiment_confidence columns")

    print()
    print("PREPROCESSING COMPLETE!")
    print(f"Final dataset: {len(df):,} rows, {len(df.columns)} columns")
    print()

    # Generate evaluation metrics
    print("COMPUTING EVALUATION METRICS")
    print("-" * 40)

    dataset_analyzer = DatasetAnalyzer(df)
    metrics = dataset_analyzer.compute_all_metrics()

    # Print summary report
    report = dataset_analyzer.generate_summary_report()
    print(report)
    print()

    # Generate visualizations
    if MATPLOTLIB_AVAILABLE:
        print("GENERATING VISUALIZATIONS")
        print("-" * 40)

        output_dir = PROJECT_ROOT / "output" / "charts"
        output_dir.mkdir(parents=True, exist_ok=True)

        visualizer = DatasetVisualizer(df, str(output_dir))
        charts = visualizer.generate_all_charts()

        print(f"Charts saved to: {output_dir}")
        print()
        for name, path in charts.items():
            if path and 'error' not in str(path):
                print(f"  - {name}: {Path(path).name}")

        print()
        print("=" * 60)
        print(f"DONE! Open the charts folder to view visualizations:")
        print(f"  {output_dir}")
        print("=" * 60)
    else:
        print("matplotlib not available - skipping visualizations")

    # Save processed dataset
    output_file = PROJECT_ROOT / "output" / "processed_dataset.xlsx"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_file, index=False)
    print(f"\nProcessed dataset saved to: {output_file}")


if __name__ == '__main__':
    main()
