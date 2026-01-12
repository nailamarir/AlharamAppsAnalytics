"""
Dataset Analyzer for AlHaram Analytics.

Computes evaluation metrics and generates visualizations for
the preprocessed app review dataset.

Author: AlHaram Analytics Team
"""

import os
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import pandas as pd
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False


class DatasetAnalyzer:
    """
    Analyze preprocessed dataset and compute evaluation metrics.

    Computes:
    - Data quality metrics (completeness, uniqueness)
    - Text statistics (length distributions, language breakdown)
    - Temporal patterns (period distribution)
    - Sentiment distribution
    - Service category breakdown
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize analyzer with a DataFrame.

        Args:
            df: Preprocessed DataFrame to analyze
        """
        self.df = df
        self.metrics = {}

    def compute_all_metrics(self) -> Dict[str, Any]:
        """Compute all evaluation metrics."""
        self.metrics = {
            'overview': self._compute_overview(),
            'data_quality': self._compute_data_quality(),
            'text_statistics': self._compute_text_statistics(),
            'language_distribution': self._compute_language_distribution(),
            'sentiment_distribution': self._compute_sentiment_distribution(),
            'period_distribution': self._compute_period_distribution(),
            'service_distribution': self._compute_service_distribution(),
            'device_distribution': self._compute_device_distribution(),
            'preprocessing_impact': self._compute_preprocessing_impact(),
        }
        return self.metrics

    def _compute_overview(self) -> Dict[str, Any]:
        """Compute dataset overview."""
        return {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'memory_usage_mb': round(self.df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
            'columns': list(self.df.columns),
        }

    def _compute_data_quality(self) -> Dict[str, Any]:
        """Compute data quality metrics."""
        total_cells = len(self.df) * len(self.df.columns)
        missing_cells = self.df.isnull().sum().sum()

        # Per-column missing values
        missing_by_column = {}
        for col in self.df.columns:
            missing = self.df[col].isnull().sum()
            if missing > 0:
                missing_by_column[col] = {
                    'count': int(missing),
                    'percentage': round(100 * missing / len(self.df), 2)
                }

        # Calculate duplicates only on hashable columns (exclude lists, dicts)
        hashable_cols = []
        for col in self.df.columns:
            try:
                sample = self.df[col].dropna().head(10)
                if len(sample) > 0:
                    first_val = sample.iloc[0]
                    if not isinstance(first_val, (list, dict)):
                        hashable_cols.append(col)
            except:
                pass

        try:
            if hashable_cols:
                dup_count = int(self.df[hashable_cols].duplicated().sum())
                dup_rate = round(100 * dup_count / len(self.df), 2)
            else:
                dup_count = 0
                dup_rate = 0.0
        except:
            dup_count = 0
            dup_rate = 0.0

        return {
            'completeness_rate': round(100 * (1 - missing_cells / total_cells), 2),
            'total_missing_values': int(missing_cells),
            'missing_by_column': missing_by_column,
            'duplicate_rows': dup_count,
            'duplicate_rate': dup_rate,
        }

    def _compute_text_statistics(self) -> Dict[str, Any]:
        """Compute text-related statistics."""
        stats = {}

        # Original text stats
        if 'Review Text' in self.df.columns:
            text_lengths = self.df['Review Text'].fillna('').str.len()
            word_counts = self.df['Review Text'].fillna('').str.split().str.len()

            stats['original_text'] = {
                'avg_length_chars': round(text_lengths.mean(), 1),
                'median_length_chars': round(text_lengths.median(), 1),
                'max_length_chars': int(text_lengths.max()),
                'avg_word_count': round(word_counts.mean(), 1),
                'median_word_count': round(word_counts.median(), 1),
            }

        # Clean text stats
        if 'clean_text' in self.df.columns:
            clean_lengths = self.df['clean_text'].fillna('').str.len()
            stats['clean_text'] = {
                'avg_length_chars': round(clean_lengths.mean(), 1),
                'reduction_rate': round(100 * (1 - clean_lengths.mean() / text_lengths.mean()), 2) if 'Review Text' in self.df.columns else 0,
            }

        # Text feature stats
        if 'text_word_count' in self.df.columns:
            stats['text_features'] = {
                'avg_word_count': round(self.df['text_word_count'].mean(), 1),
                'avg_arabic_ratio': round(self.df['text_arabic_ratio'].mean() * 100, 1) if 'text_arabic_ratio' in self.df.columns else None,
                'avg_lexical_diversity': round(self.df['text_lexical_diversity'].mean(), 3) if 'text_lexical_diversity' in self.df.columns else None,
            }

        # Emoji stats
        if 'emoji_count' in self.df.columns:
            stats['emojis'] = {
                'total_emojis': int(self.df['emoji_count'].sum()),
                'reviews_with_emojis': int((self.df['emoji_count'] > 0).sum()),
                'emoji_rate': round(100 * (self.df['emoji_count'] > 0).sum() / len(self.df), 2),
            }

        # URL stats
        if 'has_urls' in self.df.columns:
            stats['urls'] = {
                'reviews_with_urls': int(self.df['has_urls'].sum()),
                'url_rate': round(100 * self.df['has_urls'].sum() / len(self.df), 2),
            }

        return stats

    def _compute_language_distribution(self) -> Dict[str, Any]:
        """Compute language distribution."""
        if 'language' not in self.df.columns:
            return {}

        lang_counts = self.df['language'].value_counts()
        total = len(self.df)

        return {
            'distribution': {
                lang: {
                    'count': int(count),
                    'percentage': round(100 * count / total, 2)
                }
                for lang, count in lang_counts.items()
            },
            'primary_language': lang_counts.index[0] if len(lang_counts) > 0 else None,
        }

    def _compute_sentiment_distribution(self) -> Dict[str, Any]:
        """Compute sentiment distribution."""
        if 'sentiment' not in self.df.columns:
            return {}

        sent_counts = self.df['sentiment'].value_counts()
        total = len(self.df)

        result = {
            'distribution': {
                sent: {
                    'count': int(count),
                    'percentage': round(100 * count / total, 2)
                }
                for sent, count in sent_counts.items()
            }
        }

        if 'sentiment_score' in self.df.columns:
            result['score_statistics'] = {
                'mean': round(self.df['sentiment_score'].mean(), 4),
                'std': round(self.df['sentiment_score'].std(), 4),
                'median': round(self.df['sentiment_score'].median(), 4),
            }

        if 'sentiment_confidence' in self.df.columns:
            result['confidence_statistics'] = {
                'mean': round(self.df['sentiment_confidence'].mean(), 4),
                'high_confidence_rate': round(100 * (self.df['sentiment_confidence'] > 0.8).sum() / total, 2),
            }

        return result

    def _compute_period_distribution(self) -> Dict[str, Any]:
        """Compute period/event distribution."""
        if 'period' not in self.df.columns:
            return {}

        period_counts = self.df['period'].value_counts()
        total = len(self.df)

        return {
            'distribution': {
                period: {
                    'count': int(count),
                    'percentage': round(100 * count / total, 2)
                }
                for period, count in period_counts.items()
            }
        }

    def _compute_service_distribution(self) -> Dict[str, Any]:
        """Compute service type distribution."""
        if 'Service_Type' not in self.df.columns:
            return {}

        service_counts = self.df['Service_Type'].value_counts()
        total = len(self.df)

        return {
            'distribution': {
                service: {
                    'count': int(count),
                    'percentage': round(100 * count / total, 2)
                }
                for service, count in service_counts.items()
            }
        }

    def _compute_device_distribution(self) -> Dict[str, Any]:
        """Compute device type distribution."""
        if 'Device Type' not in self.df.columns:
            return {}

        device_counts = self.df['Device Type'].value_counts()
        total = len(self.df)

        return {
            'distribution': {
                device: {
                    'count': int(count),
                    'percentage': round(100 * count / total, 2)
                }
                for device, count in device_counts.items()
            }
        }

    def _compute_preprocessing_impact(self) -> Dict[str, Any]:
        """Compute preprocessing impact metrics."""
        impact = {}

        # Username cleaning impact
        if 'User Name' in self.df.columns and 'clean_name' in self.df.columns:
            original_unique = self.df['User Name'].nunique()
            cleaned_unique = self.df['clean_name'].nunique()
            anonymous_count = (self.df['clean_name'] == 'Anonymous').sum()

            impact['username_cleaning'] = {
                'original_unique': int(original_unique),
                'cleaned_unique': int(cleaned_unique),
                'consolidation_rate': round(100 * (1 - cleaned_unique / original_unique), 2) if original_unique > 0 else 0,
                'anonymous_count': int(anonymous_count),
                'anonymous_rate': round(100 * anonymous_count / len(self.df), 2),
            }

        # Text cleaning impact
        if 'has_diacritics' in self.df.columns:
            impact['diacritics'] = {
                'reviews_with_diacritics': int(self.df['has_diacritics'].sum()),
                'diacritics_rate': round(100 * self.df['has_diacritics'].sum() / len(self.df), 2),
            }

        if 'has_elongation' in self.df.columns:
            impact['elongation'] = {
                'reviews_with_elongation': int(self.df['has_elongation'].sum()),
                'elongation_rate': round(100 * self.df['has_elongation'].sum() / len(self.df), 2),
            }

        return impact

    def generate_summary_report(self) -> str:
        """Generate a text summary report."""
        if not self.metrics:
            self.compute_all_metrics()

        lines = [
            "=" * 60,
            "ALHARAM ANALYTICS - DATASET EVALUATION REPORT",
            "=" * 60,
            "",
            "DATASET OVERVIEW",
            "-" * 40,
            f"Total Reviews: {self.metrics['overview']['total_rows']:,}",
            f"Total Columns: {self.metrics['overview']['total_columns']}",
            f"Memory Usage: {self.metrics['overview']['memory_usage_mb']} MB",
            "",
            "DATA QUALITY METRICS",
            "-" * 40,
            f"Completeness Rate: {self.metrics['data_quality']['completeness_rate']}%",
            f"Duplicate Rows: {self.metrics['data_quality']['duplicate_rows']} ({self.metrics['data_quality']['duplicate_rate']}%)",
            "",
        ]

        # Language distribution
        if self.metrics.get('language_distribution'):
            lines.extend([
                "LANGUAGE DISTRIBUTION",
                "-" * 40,
            ])
            for lang, data in self.metrics['language_distribution']['distribution'].items():
                lines.append(f"  {lang}: {data['count']:,} ({data['percentage']}%)")
            lines.append("")

        # Sentiment distribution
        if self.metrics.get('sentiment_distribution'):
            lines.extend([
                "SENTIMENT DISTRIBUTION",
                "-" * 40,
            ])
            for sent, data in self.metrics['sentiment_distribution']['distribution'].items():
                lines.append(f"  {sent}: {data['count']:,} ({data['percentage']}%)")
            if 'score_statistics' in self.metrics['sentiment_distribution']:
                stats = self.metrics['sentiment_distribution']['score_statistics']
                lines.append(f"  Average Score: {stats['mean']:.4f}")
            lines.append("")

        # Period distribution
        if self.metrics.get('period_distribution'):
            lines.extend([
                "PERIOD DISTRIBUTION",
                "-" * 40,
            ])
            for period, data in self.metrics['period_distribution']['distribution'].items():
                lines.append(f"  {period}: {data['count']:,} ({data['percentage']}%)")
            lines.append("")

        # Text statistics
        if self.metrics.get('text_statistics'):
            lines.extend([
                "TEXT STATISTICS",
                "-" * 40,
            ])
            if 'original_text' in self.metrics['text_statistics']:
                stats = self.metrics['text_statistics']['original_text']
                lines.append(f"  Avg Word Count: {stats['avg_word_count']}")
                lines.append(f"  Avg Length (chars): {stats['avg_length_chars']}")
            if 'text_features' in self.metrics['text_statistics']:
                stats = self.metrics['text_statistics']['text_features']
                if stats.get('avg_arabic_ratio'):
                    lines.append(f"  Avg Arabic Ratio: {stats['avg_arabic_ratio']}%")
                if stats.get('avg_lexical_diversity'):
                    lines.append(f"  Avg Lexical Diversity: {stats['avg_lexical_diversity']}")
            lines.append("")

        lines.extend([
            "=" * 60,
            "End of Report",
            "=" * 60,
        ])

        return "\n".join(lines)


class DatasetVisualizer:
    """Generate visualizations for the preprocessed dataset."""

    # Color palettes
    COLORS = {
        'primary': '#6366f1',
        'secondary': '#8b5cf6',
        'success': '#10b981',
        'warning': '#f59e0b',
        'danger': '#ef4444',
        'info': '#06b6d4',
    }

    SENTIMENT_COLORS = {
        'positive': '#10b981',
        'neutral': '#6b7280',
        'negative': '#ef4444',
    }

    LANGUAGE_COLORS = {
        'Arabic': '#6366f1',
        'English': '#8b5cf6',
        'Mixed': '#f59e0b',
        'Unknown': '#9ca3af',
    }

    def __init__(self, df: pd.DataFrame, output_dir: str = 'static/charts'):
        """
        Initialize visualizer.

        Args:
            df: DataFrame to visualize
            output_dir: Directory to save chart images
        """
        self.df = df
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if MATPLOTLIB_AVAILABLE:
            plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'ggplot')

    def generate_all_charts(self) -> Dict[str, str]:
        """Generate all charts and return paths."""
        if not MATPLOTLIB_AVAILABLE:
            return {'error': 'matplotlib not available'}

        charts = {}

        # Language distribution pie chart
        if 'language' in self.df.columns:
            charts['language_pie'] = self._create_language_pie()

        # Sentiment distribution bar chart
        if 'sentiment' in self.df.columns:
            charts['sentiment_bar'] = self._create_sentiment_bar()

        # Period distribution bar chart
        if 'period' in self.df.columns:
            charts['period_bar'] = self._create_period_bar()

        # Service type distribution
        if 'Service_Type' in self.df.columns:
            charts['service_pie'] = self._create_service_pie()

        # Word count distribution
        if 'text_word_count' in self.df.columns:
            charts['word_count_hist'] = self._create_word_count_histogram()

        # Arabic ratio distribution
        if 'text_arabic_ratio' in self.df.columns:
            charts['arabic_ratio_hist'] = self._create_arabic_ratio_histogram()

        # Device distribution
        if 'Device Type' in self.df.columns:
            charts['device_pie'] = self._create_device_pie()

        # Sentiment by period (if both exist)
        if 'sentiment' in self.df.columns and 'period' in self.df.columns:
            charts['sentiment_by_period'] = self._create_sentiment_by_period()

        return charts

    def _create_language_pie(self) -> str:
        """Create language distribution pie chart."""
        fig, ax = plt.subplots(figsize=(8, 6))

        counts = self.df['language'].value_counts()
        colors = [self.LANGUAGE_COLORS.get(lang, '#9ca3af') for lang in counts.index]

        wedges, texts, autotexts = ax.pie(
            counts.values,
            labels=counts.index,
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            explode=[0.02] * len(counts)
        )

        ax.set_title('Language Distribution', fontsize=14, fontweight='bold')

        path = self.output_dir / 'language_distribution.png'
        plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

        return str(path)

    def _create_sentiment_bar(self) -> str:
        """Create sentiment distribution bar chart."""
        fig, ax = plt.subplots(figsize=(8, 5))

        counts = self.df['sentiment'].value_counts()
        colors = [self.SENTIMENT_COLORS.get(sent, '#6b7280') for sent in counts.index]

        bars = ax.bar(counts.index, counts.values, color=colors, edgecolor='white', linewidth=1.5)

        # Add value labels
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                   f'{val:,}', ha='center', va='bottom', fontweight='bold')

        ax.set_xlabel('Sentiment', fontsize=12)
        ax.set_ylabel('Number of Reviews', fontsize=12)
        ax.set_title('Sentiment Distribution', fontsize=14, fontweight='bold')

        path = self.output_dir / 'sentiment_distribution.png'
        plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

        return str(path)

    def _create_period_bar(self) -> str:
        """Create period distribution bar chart."""
        fig, ax = plt.subplots(figsize=(10, 5))

        counts = self.df['period'].value_counts()

        bars = ax.barh(counts.index, counts.values, color=self.COLORS['primary'], edgecolor='white')

        # Add value labels
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height()/2,
                   f'{val:,}', ha='left', va='center')

        ax.set_xlabel('Number of Reviews', fontsize=12)
        ax.set_ylabel('Period', fontsize=12)
        ax.set_title('Reviews by Islamic Calendar Period', fontsize=14, fontweight='bold')

        path = self.output_dir / 'period_distribution.png'
        plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

        return str(path)

    def _create_service_pie(self) -> str:
        """Create service type distribution pie chart."""
        fig, ax = plt.subplots(figsize=(9, 6))

        counts = self.df['Service_Type'].value_counts()
        colors = [self.COLORS['primary'], self.COLORS['secondary'],
                  self.COLORS['success'], self.COLORS['warning'], self.COLORS['info']]

        wedges, texts, autotexts = ax.pie(
            counts.values,
            labels=counts.index,
            autopct='%1.1f%%',
            colors=colors[:len(counts)],
            startangle=90,
        )

        ax.set_title('Service Type Distribution', fontsize=14, fontweight='bold')

        path = self.output_dir / 'service_distribution.png'
        plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

        return str(path)

    def _create_word_count_histogram(self) -> str:
        """Create word count distribution histogram."""
        fig, ax = plt.subplots(figsize=(8, 5))

        data = self.df['text_word_count'].dropna()
        data = data[data <= data.quantile(0.95)]  # Remove outliers

        ax.hist(data, bins=30, color=self.COLORS['primary'], edgecolor='white', alpha=0.8)

        ax.axvline(data.mean(), color=self.COLORS['danger'], linestyle='--',
                  label=f'Mean: {data.mean():.1f}')
        ax.axvline(data.median(), color=self.COLORS['warning'], linestyle='--',
                  label=f'Median: {data.median():.1f}')

        ax.set_xlabel('Word Count', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Review Word Count Distribution', fontsize=14, fontweight='bold')
        ax.legend()

        path = self.output_dir / 'word_count_distribution.png'
        plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

        return str(path)

    def _create_arabic_ratio_histogram(self) -> str:
        """Create Arabic ratio distribution histogram."""
        fig, ax = plt.subplots(figsize=(8, 5))

        data = self.df['text_arabic_ratio'].dropna()

        ax.hist(data, bins=20, color=self.COLORS['secondary'], edgecolor='white', alpha=0.8)

        ax.set_xlabel('Arabic Character Ratio', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Arabic Content Ratio Distribution', fontsize=14, fontweight='bold')

        path = self.output_dir / 'arabic_ratio_distribution.png'
        plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

        return str(path)

    def _create_device_pie(self) -> str:
        """Create device type distribution pie chart."""
        fig, ax = plt.subplots(figsize=(7, 5))

        counts = self.df['Device Type'].value_counts()
        colors = ['#007AFF', '#3DDC84', '#9ca3af']  # iOS blue, Android green, Other gray

        wedges, texts, autotexts = ax.pie(
            counts.values,
            labels=counts.index,
            autopct='%1.1f%%',
            colors=colors[:len(counts)],
            startangle=90,
        )

        ax.set_title('Device Type Distribution', fontsize=14, fontweight='bold')

        path = self.output_dir / 'device_distribution.png'
        plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

        return str(path)

    def _create_sentiment_by_period(self) -> str:
        """Create sentiment distribution by period stacked bar chart."""
        fig, ax = plt.subplots(figsize=(12, 6))

        # Create cross-tabulation
        cross = pd.crosstab(self.df['period'], self.df['sentiment'], normalize='index') * 100

        # Reorder columns
        cols = ['negative', 'neutral', 'positive']
        cols = [c for c in cols if c in cross.columns]
        cross = cross[cols]

        # Create stacked bar
        cross.plot(kind='bar', stacked=True, ax=ax,
                  color=[self.SENTIMENT_COLORS.get(c, '#6b7280') for c in cols],
                  edgecolor='white', linewidth=0.5)

        ax.set_xlabel('Period', fontsize=12)
        ax.set_ylabel('Percentage (%)', fontsize=12)
        ax.set_title('Sentiment Distribution by Period', fontsize=14, fontweight='bold')
        ax.legend(title='Sentiment', bbox_to_anchor=(1.02, 1), loc='upper left')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

        path = self.output_dir / 'sentiment_by_period.png'
        plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

        return str(path)


def analyze_dataset(df: pd.DataFrame, output_dir: str = None) -> Tuple[Dict, Dict]:
    """
    Convenience function to analyze dataset and generate visualizations.

    Args:
        df: DataFrame to analyze
        output_dir: Directory for chart outputs

    Returns:
        Tuple of (metrics dict, chart paths dict)
    """
    analyzer = DatasetAnalyzer(df)
    metrics = analyzer.compute_all_metrics()

    charts = {}
    if MATPLOTLIB_AVAILABLE and output_dir:
        visualizer = DatasetVisualizer(df, output_dir)
        charts = visualizer.generate_all_charts()

    return metrics, charts
