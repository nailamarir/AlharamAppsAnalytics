#!/usr/bin/env python3
"""
Script to run the preprocessing pipeline.

Usage:
    python scripts/run_preprocessing.py --input data/raw/reviews.xlsx --output data/processed/reviews_clean.xlsx
    python scripts/run_preprocessing.py --input data/raw/reviews.xlsx --output data/processed/reviews_clean.xlsx --gender
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from alharam_analytics.pipeline import PreprocessingPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Run AlHaram Analytics preprocessing pipeline"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input data file (xlsx, csv, parquet)"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path to output file"
    )
    parser.add_argument(
        "--gender",
        action="store_true",
        help="Include gender prediction (requires HuggingFace transformers)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress messages"
    )

    args = parser.parse_args()

    # Run pipeline
    pipeline = PreprocessingPipeline(
        include_gender_prediction=args.gender,
        verbose=not args.quiet
    )

    df = pipeline.run(args.input)
    pipeline.save(df, args.output)

    print(f"\nProcessed {len(df)} rows")
    print(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main()
