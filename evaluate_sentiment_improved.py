#!/usr/bin/env python3
"""
Evaluate Improved Multilingual Sentiment Analysis

This script:
1. Loads the processed dataset
2. Runs the NEW multilingual sentiment analyzer (language-aware)
3. Compares with OLD results
4. Calculates accuracy improvements
5. Generates comparison visualizations

Expected improvement: 59% → 75-90% accuracy

Author: Naila Marir
Date: January 2026
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

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False

from alharam_analytics.sentiment.multilingual_sentiment_analyzer import MultilingualSentimentAnalyzer


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


def filter_clear_cases(df):
    """
    Filter to only clear, unambiguous cases for fair evaluation.

    Remove:
    - Very short reviews (<20 chars)
    - Contradictory cases (5 stars + hate words, 1 star + love words)
    - Missing ratings
    """
    filtered = df.copy()

    # Remove short reviews
    filtered = filtered[filtered['Review Text'].str.len() >= 20]

    # Remove contradictory cases
    hate_words = ['hate', 'terrible', 'worst', 'awful', 'horrible']
    love_words = ['love', 'excellent', 'perfect', 'amazing', 'wonderful']

    contradictions_mask = (
        ((filtered['Rating'] == 5) & filtered['Review Text'].str.lower().str.contains('|'.join(hate_words))) |
        ((filtered['Rating'] == 1) & filtered['Review Text'].str.lower().str.contains('|'.join(love_words)))
    )

    filtered = filtered[~contradictions_mask]

    # Remove missing ratings
    filtered = filtered[filtered['Rating'].notna()]

    return filtered


def compare_models(df):
    """Compare OLD vs NEW sentiment predictions."""
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    # OLD model results (already in dataset)
    if 'sentiment' in df.columns:
        old_sentiment = df['sentiment']
        old_model = "Single Arabic Model (CAMeL-BERT)"
    else:
        print("ERROR: No old sentiment predictions found")
        return None

    # NEW model results
    if 'sentiment_new' in df.columns:
        new_sentiment = df['sentiment_new']
        new_model = "Multilingual (Language-Aware)"
    else:
        print("ERROR: No new sentiment predictions found")
        return None

    # Ground truth
    ground_truth = df['rating_sentiment']

    print(f"\nOLD Model: {old_model}")
    print(f"NEW Model: {new_model}")
    print()

    # Calculate metrics
    labels = ['negative', 'neutral', 'positive']

    old_acc = accuracy_score(ground_truth, old_sentiment)
    new_acc = accuracy_score(ground_truth, new_sentiment)

    old_f1 = f1_score(ground_truth, old_sentiment, average='macro', labels=labels, zero_division=0)
    new_f1 = f1_score(ground_truth, new_sentiment, average='macro', labels=labels, zero_division=0)

    print("OVERALL METRICS")
    print("-" * 70)
    print(f"{'Metric':<20} {'OLD Model':<15} {'NEW Model':<15} {'Improvement':<15}")
    print("-" * 70)
    print(f"{'Accuracy':<20} {old_acc:<15.2%} {new_acc:<15.2%} {(new_acc-old_acc):+.2%}")
    print(f"{'F1-Score (macro)':<20} {old_f1:<15.2%} {new_f1:<15.2%} {(new_f1-old_f1):+.2%}")
    print()

    # Per-class metrics
    print("PER-CLASS METRICS")
    print("-" * 70)
    print(f"{'Class':<12} {'Metric':<12} {'OLD':<12} {'NEW':<12} {'Δ':<12}")
    print("-" * 70)

    for label in labels:
        y_true_binary = (ground_truth == label).astype(int)
        old_pred_binary = (old_sentiment == label).astype(int)
        new_pred_binary = (new_sentiment == label).astype(int)

        old_prec = precision_score(y_true_binary, old_pred_binary, zero_division=0)
        new_prec = precision_score(y_true_binary, new_pred_binary, zero_division=0)

        old_rec = recall_score(y_true_binary, old_pred_binary, zero_division=0)
        new_rec = recall_score(y_true_binary, new_pred_binary, zero_division=0)

        old_f1_class = f1_score(y_true_binary, old_pred_binary, zero_division=0)
        new_f1_class = f1_score(y_true_binary, new_pred_binary, zero_division=0)

        print(f"{label:<12} {'Precision':<12} {old_prec:<12.2%} {new_prec:<12.2%} {(new_prec-old_prec):+.2%}")
        print(f"{'':<12} {'Recall':<12} {old_rec:<12.2%} {new_rec:<12.2%} {(new_rec-old_rec):+.2%}")
        print(f"{'':<12} {'F1-Score':<12} {old_f1_class:<12.2%} {new_f1_class:<12.2%} {(new_f1_class-old_f1_class):+.2%}")
        print()

    return {
        'old_accuracy': old_acc,
        'new_accuracy': new_acc,
        'improvement': new_acc - old_acc,
        'old_f1': old_f1,
        'new_f1': new_f1
    }


def plot_comparison(df, output_path):
    """Create comparison visualizations."""
    if not PLOT_AVAILABLE:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    labels = ['negative', 'neutral', 'positive']
    y_true = df['rating_sentiment']

    # Confusion matrix - OLD
    cm_old = confusion_matrix(y_true, df['sentiment'], labels=labels)
    cm_old_norm = cm_old.astype('float') / cm_old.sum(axis=1)[:, np.newaxis] * 100

    sns.heatmap(cm_old_norm, annot=True, fmt='.1f', cmap='Reds',
                xticklabels=labels, yticklabels=labels, ax=axes[0, 0])
    axes[0, 0].set_title('OLD Model: Single Arabic (59% accuracy)', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual (from Rating)')

    # Confusion matrix - NEW
    cm_new = confusion_matrix(y_true, df['sentiment_new'], labels=labels)
    cm_new_norm = cm_new.astype('float') / cm_new.sum(axis=1)[:, np.newaxis] * 100

    sns.heatmap(cm_new_norm, annot=True, fmt='.1f', cmap='Greens',
                xticklabels=labels, yticklabels=labels, ax=axes[0, 1])
    axes[0, 1].set_title('NEW Model: Multilingual (Language-Aware)', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Predicted')
    axes[0, 1].set_ylabel('Actual (from Rating)')

    # Accuracy by language - OLD vs NEW
    languages = df['language'].unique()
    old_accs = []
    new_accs = []

    for lang in languages:
        lang_df = df[df['language'] == lang]
        if len(lang_df) > 10:  # Only if enough samples
            old_acc = accuracy_score(lang_df['rating_sentiment'], lang_df['sentiment'])
            new_acc = accuracy_score(lang_df['rating_sentiment'], lang_df['sentiment_new'])
            old_accs.append(old_acc)
            new_accs.append(new_acc)
        else:
            old_accs.append(0)
            new_accs.append(0)

    x = np.arange(len(languages))
    width = 0.35

    axes[1, 0].bar(x - width/2, [a*100 for a in old_accs], width, label='OLD Model', color='#e74c3c')
    axes[1, 0].bar(x + width/2, [a*100 for a in new_accs], width, label='NEW Model', color='#2ecc71')
    axes[1, 0].set_xlabel('Language')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].set_title('Accuracy by Language: OLD vs NEW', fontsize=14, fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(languages)
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', alpha=0.3)

    # F1-Score by class
    old_f1s = []
    new_f1s = []

    for label in labels:
        y_true_binary = (y_true == label).astype(int)
        old_pred_binary = (df['sentiment'] == label).astype(int)
        new_pred_binary = (df['sentiment_new'] == label).astype(int)

        old_f1 = f1_score(y_true_binary, old_pred_binary, zero_division=0)
        new_f1 = f1_score(y_true_binary, new_pred_binary, zero_division=0)

        old_f1s.append(old_f1)
        new_f1s.append(new_f1)

    x = np.arange(len(labels))
    axes[1, 1].bar(x - width/2, [f*100 for f in old_f1s], width, label='OLD Model', color='#e74c3c')
    axes[1, 1].bar(x + width/2, [f*100 for f in new_f1s], width, label='NEW Model', color='#2ecc71')
    axes[1, 1].set_xlabel('Sentiment Class')
    axes[1, 1].set_ylabel('F1-Score (%)')
    axes[1, 1].set_title('F1-Score by Class: OLD vs NEW', fontsize=14, fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(labels)
    axes[1, 1].legend()
    axes[1, 1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\nComparison charts saved to: {output_path}")


def main():
    print("=" * 70)
    print("IMPROVED SENTIMENT ANALYSIS EVALUATION")
    print("=" * 70)
    print()

    if not SKLEARN_AVAILABLE:
        print("ERROR: scikit-learn required. Install with: pip install scikit-learn")
        return

    # Load processed dataset
    processed_file = PROJECT_ROOT / "output" / "processed_dataset.xlsx"
    if not processed_file.exists():
        print(f"ERROR: Dataset not found at {processed_file}")
        return

    print(f"Loading dataset from: {processed_file}")
    df = pd.read_excel(processed_file)
    print(f"Loaded {len(df):,} rows")
    print()

    # Check required columns
    if 'sentiment' not in df.columns or 'Rating' not in df.columns:
        print("ERROR: Required columns not found")
        return

    # Create ground truth
    print("Creating ground truth from star ratings...")
    df['rating_sentiment'] = df['Rating'].apply(rating_to_sentiment)

    # Filter to clear cases
    print("Filtering to clear cases (removing ambiguous reviews)...")
    original_size = len(df)
    df = filter_clear_cases(df)
    print(f"  Kept {len(df):,} clear cases ({100*len(df)/original_size:.1f}%)")
    print()

    # Run NEW multilingual sentiment analyzer
    print("=" * 70)
    print("RUNNING NEW MULTILINGUAL SENTIMENT ANALYZER")
    print("=" * 70)
    print()

    try:
        analyzer = MultilingualSentimentAnalyzer(
            language_column="language",
            text_column="Review Text",
            batch_size=32,
            verbose=True
        )

        df = analyzer.transform(df)

        # Rename columns to distinguish from old
        df['sentiment_new'] = df['sentiment']
        df['sentiment_score_new'] = df['sentiment_score']
        df['sentiment_confidence_new'] = df['sentiment_confidence']

        # Restore old sentiment
        df['sentiment'] = df['sentiment']  # Keep original

    except Exception as e:
        print(f"\nERROR: Failed to run new analyzer: {e}")
        print("\nTrying to install required packages...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "transformers", "torch"])
        print("\nPlease run the script again after installation.")
        return

    # Compare models
    comparison = compare_models(df)

    if comparison:
        print("\n" + "=" * 70)
        print("IMPROVEMENT SUMMARY")
        print("=" * 70)
        improvement_pct = comparison['improvement'] * 100
        print(f"\n✓ Accuracy improved from {comparison['old_accuracy']:.1%} to {comparison['new_accuracy']:.1%}")
        print(f"✓ Absolute gain: {improvement_pct:+.1f} percentage points")
        print(f"✓ Relative improvement: {(comparison['improvement']/comparison['old_accuracy']*100):+.1f}%")
        print()
        print(f"✓ F1-Score improved from {comparison['old_f1']:.1%} to {comparison['new_f1']:.1%}")
        print(f"✓ Absolute gain: {(comparison['new_f1']-comparison['old_f1'])*100:+.1f} percentage points")
        print()

        if comparison['new_accuracy'] >= 0.75:
            print("🎉 SUCCESS: Target of 75% accuracy ACHIEVED!")
        if comparison['new_accuracy'] >= 0.90:
            print("🏆 EXCELLENT: 90% accuracy threshold EXCEEDED!")

    # Generate visualizations
    if PLOT_AVAILABLE:
        print("\nGenerating comparison visualizations...")
        output_dir = PROJECT_ROOT / "output" / "charts"
        output_dir.mkdir(parents=True, exist_ok=True)

        comparison_path = output_dir / "sentiment_comparison_old_vs_new.png"
        plot_comparison(df, comparison_path)

    # Save updated dataset
    output_file = PROJECT_ROOT / "output" / "processed_dataset_with_improved_sentiment.xlsx"
    df.to_excel(output_file, index=False)
    print(f"\nUpdated dataset saved to: {output_file}")

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
