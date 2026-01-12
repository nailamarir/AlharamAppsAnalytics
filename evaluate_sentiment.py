#!/usr/bin/env python3
"""
Evaluate Sentiment Analysis Accuracy using Star Ratings as Ground Truth.

This script:
1. Loads the processed dataset
2. Maps star ratings to sentiment labels (ground truth proxy)
3. Compares with model predictions
4. Calculates accuracy, precision, recall, F1-score
5. Generates confusion matrix visualization

Usage:
    python evaluate_sentiment.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
import numpy as np

try:
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        classification_report,
        confusion_matrix
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not installed. Install with: pip install scikit-learn")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False


def rating_to_sentiment(rating):
    """Convert star rating to sentiment label."""
    if pd.isna(rating):
        return None
    rating = float(rating)
    if rating <= 2:
        return 'negative'
    elif rating == 3:
        return 'neutral'
    else:  # 4-5
        return 'positive'


def calculate_metrics(y_true, y_pred, labels=['negative', 'neutral', 'positive']):
    """Calculate all evaluation metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', labels=labels, zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', labels=labels, zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', labels=labels, zero_division=0),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', labels=labels, zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', labels=labels, zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', labels=labels, zero_division=0),
    }

    # Per-class metrics
    for label in labels:
        y_true_binary = (y_true == label).astype(int)
        y_pred_binary = (y_pred == label).astype(int)
        metrics[f'precision_{label}'] = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        metrics[f'recall_{label}'] = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        metrics[f'f1_{label}'] = f1_score(y_true_binary, y_pred_binary, zero_division=0)

    return metrics


def plot_confusion_matrix(y_true, y_pred, labels, output_path):
    """Create and save confusion matrix visualization."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=axes[0])
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual (from Rating)')
    axes[0].set_title('Confusion Matrix (Counts)')

    # Normalized (percentages)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    sns.heatmap(cm_normalized, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=axes[1])
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual (from Rating)')
    axes[1].set_title('Confusion Matrix (Normalized %)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return cm


def main():
    print("=" * 60)
    print("SENTIMENT ANALYSIS EVALUATION")
    print("=" * 60)
    print()

    if not SKLEARN_AVAILABLE:
        print("ERROR: scikit-learn is required for evaluation.")
        print("Install with: pip install scikit-learn")
        return

    # Load processed dataset
    processed_file = PROJECT_ROOT / "output" / "processed_dataset.xlsx"
    if not processed_file.exists():
        print(f"ERROR: Processed dataset not found at {processed_file}")
        print("Run generate_visualizations.py first to create the processed dataset.")
        return

    print(f"Loading dataset from: {processed_file}")
    df = pd.read_excel(processed_file)
    print(f"Loaded {len(df):,} rows")
    print()

    # Check required columns
    if 'sentiment' not in df.columns:
        print("ERROR: 'sentiment' column not found. Run sentiment analysis first.")
        return

    if 'Rating' not in df.columns:
        print("ERROR: 'Rating' column not found. Cannot use star ratings as ground truth.")
        return

    # Create ground truth from ratings
    print("CREATING GROUND TRUTH FROM STAR RATINGS")
    print("-" * 40)
    print("Mapping: 1-2 stars → negative, 3 stars → neutral, 4-5 stars → positive")
    print()

    df['rating_sentiment'] = df['Rating'].apply(rating_to_sentiment)

    # Filter rows with valid ratings and predictions
    valid_mask = df['rating_sentiment'].notna() & df['sentiment'].notna()
    df_valid = df[valid_mask].copy()

    print(f"Valid samples for evaluation: {len(df_valid):,} ({100*len(df_valid)/len(df):.1f}%)")
    print()

    # Get predictions and ground truth
    y_true = df_valid['rating_sentiment']
    y_pred = df_valid['sentiment']

    labels = ['negative', 'neutral', 'positive']

    # Distribution comparison
    print("DISTRIBUTION COMPARISON")
    print("-" * 40)
    print("\nGround Truth (from Ratings):")
    gt_dist = y_true.value_counts()
    for label in labels:
        count = gt_dist.get(label, 0)
        pct = 100 * count / len(y_true)
        print(f"  {label:10s}: {count:,} ({pct:.1f}%)")

    print("\nModel Predictions:")
    pred_dist = y_pred.value_counts()
    for label in labels:
        count = pred_dist.get(label, 0)
        pct = 100 * count / len(y_pred)
        print(f"  {label:10s}: {count:,} ({pct:.1f}%)")
    print()

    # Calculate metrics
    print("EVALUATION METRICS")
    print("-" * 40)

    metrics = calculate_metrics(y_true, y_pred, labels)

    print(f"\nOverall Accuracy: {metrics['accuracy']:.2%}")
    print()
    print("Macro-averaged Metrics (equal weight to each class):")
    print(f"  Precision: {metrics['precision_macro']:.2%}")
    print(f"  Recall:    {metrics['recall_macro']:.2%}")
    print(f"  F1-Score:  {metrics['f1_macro']:.2%}")
    print()
    print("Weighted-averaged Metrics (weighted by class frequency):")
    print(f"  Precision: {metrics['precision_weighted']:.2%}")
    print(f"  Recall:    {metrics['recall_weighted']:.2%}")
    print(f"  F1-Score:  {metrics['f1_weighted']:.2%}")
    print()

    # Per-class metrics
    print("Per-Class Metrics:")
    print(f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 48)
    for label in labels:
        print(f"{label:<12} {metrics[f'precision_{label}']:<12.2%} {metrics[f'recall_{label}']:<12.2%} {metrics[f'f1_{label}']:<12.2%}")
    print()

    # Full classification report
    print("DETAILED CLASSIFICATION REPORT")
    print("-" * 40)
    print(classification_report(y_true, y_pred, labels=labels, digits=4))

    # Generate confusion matrix
    if PLOT_AVAILABLE:
        print("GENERATING CONFUSION MATRIX")
        print("-" * 40)
        output_dir = PROJECT_ROOT / "output" / "charts"
        output_dir.mkdir(parents=True, exist_ok=True)

        cm_path = output_dir / "sentiment_confusion_matrix.png"
        cm = plot_confusion_matrix(y_true, y_pred, labels, cm_path)
        print(f"Confusion matrix saved to: {cm_path}")
        print()

        print("Confusion Matrix (Counts):")
        print(f"{'':12s} {'Pred_neg':>10s} {'Pred_neu':>10s} {'Pred_pos':>10s}")
        for i, row_label in enumerate(labels):
            print(f"{row_label:<12s} {cm[i,0]:>10,d} {cm[i,1]:>10,d} {cm[i,2]:>10,d}")

    print()
    print("=" * 60)
    print("INTERPRETATION NOTES")
    print("=" * 60)
    print("""
1. Ground truth is derived from star ratings (proxy, not perfect):
   - Users may give 5 stars with negative text (or vice versa)
   - Rating reflects overall satisfaction, not text sentiment

2. The sentiment model analyzes the TEXT content, while ratings
   reflect the user's ACTION (choosing stars).

3. Disagreements may indicate:
   - Sarcasm or mixed sentiment in reviews
   - Users who rate based on factors not mentioned in text
   - Model limitations in understanding context

4. For true accuracy, you would need human-annotated sentiment
   labels on a sample of reviews.
""")

    # Save metrics to file
    metrics_file = PROJECT_ROOT / "output" / "sentiment_evaluation_metrics.txt"
    with open(metrics_file, 'w') as f:
        f.write("SENTIMENT ANALYSIS EVALUATION METRICS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total samples evaluated: {len(df_valid):,}\n\n")
        f.write(f"Overall Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']:.2%})\n\n")
        f.write("Macro-averaged:\n")
        f.write(f"  Precision: {metrics['precision_macro']:.4f}\n")
        f.write(f"  Recall: {metrics['recall_macro']:.4f}\n")
        f.write(f"  F1-Score: {metrics['f1_macro']:.4f}\n\n")
        f.write("Weighted-averaged:\n")
        f.write(f"  Precision: {metrics['precision_weighted']:.4f}\n")
        f.write(f"  Recall: {metrics['recall_weighted']:.4f}\n")
        f.write(f"  F1-Score: {metrics['f1_weighted']:.4f}\n\n")
        f.write("Per-class:\n")
        for label in labels:
            f.write(f"  {label}: P={metrics[f'precision_{label}']:.4f}, R={metrics[f'recall_{label}']:.4f}, F1={metrics[f'f1_{label}']:.4f}\n")
        f.write("\n")
        f.write(classification_report(y_true, y_pred, labels=labels, digits=4))

    print(f"\nMetrics saved to: {metrics_file}")


if __name__ == '__main__':
    main()
