"""
AlHaram Analytics Web Application
Interactive preprocessing pipeline with before/after visualization
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from flask import Flask, render_template, request, jsonify, send_file, session
from werkzeug.utils import secure_filename
import pandas as pd

from alharam_analytics.preprocessing import UsernamePreprocessor, LanguageDetector, TextCleaner
from alharam_analytics.feature_engineering.period_tagger import PeriodTagger
from alharam_analytics.feature_engineering.device_mapper import DeviceTypeMapper
from alharam_analytics.feature_engineering.app_name_normalizer import AppNameNormalizer
from alharam_analytics.feature_engineering.service_classifier import ServiceClassifier
from alharam_analytics.feature_engineering.text_feature_extractor import TextFeatureExtractor
from alharam_analytics.sentiment import SentimentAnalyzer, SimpleSentimentAnalyzer, TRANSFORMERS_AVAILABLE
from alharam_analytics.analytics import DatasetAnalyzer, DatasetVisualizer, MATPLOTLIB_AVAILABLE

app = Flask(__name__)
app.secret_key = 'alharam-analytics-secret-key-2024'
app.config['UPLOAD_FOLDER'] = Path(__file__).parent / 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max

# Ensure upload folder exists
app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)

# Store dataframes in memory (for demo - use database in production)
data_store = {}


def clean_for_json(obj):
    """Clean data for JSON serialization (handle NaN, NaT, int64, etc.)."""
    import math
    import numpy as np

    if obj is None:
        return None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    if pd.isna(obj):
        return None
    return obj


def get_sample_data(df, n=10):
    """Get sample data for preview."""
    sample = df.head(n).copy()
    # Convert datetime columns to string for JSON serialization
    for col in sample.columns:
        if pd.api.types.is_datetime64_any_dtype(sample[col]):
            sample[col] = sample[col].astype(str)
    # Replace NaN with None for JSON
    sample = sample.fillna('')
    result = sample.to_dict(orient='records')
    return clean_for_json(result)


def get_column_stats(df, column):
    """Get statistics for a column."""
    if column not in df.columns:
        return {}

    col = df[column]
    stats = {
        'total': int(len(col)),
        'unique': int(col.nunique()),
        'missing': int(col.isna().sum()),
        'missing_pct': round(float(col.isna().sum()) / len(col) * 100, 2)
    }

    if col.dtype == 'object':
        value_counts = col.value_counts().head(10).to_dict()
        stats['top_values'] = {str(k): int(v) for k, v in value_counts.items()}

    return clean_for_json(stats)


PIPELINE_STEPS = [
    {
        'id': 'text_clean',
        'name': 'Text Cleaning',
        'icon': 'fa-broom',
        'color': '#14b8a6',
        'description': 'Clean Arabic text while extracting URLs, emojis, hashtags, and mentions to separate columns.',
        'details': [
            'Extract URLs, emails, hashtags, mentions to separate columns',
            'Extract emojis with counts (preserved for sentiment analysis)',
            'Normalize Arabic characters (أإآ → ا, ى → ي)',
            'Strip diacritics (tashkeel) while flagging their presence',
            'Collapse repeated characters (شككككرا → شكرا)',
            'Remove zero-width characters and normalize whitespace'
        ],
        'input_col': 'Review Text',
        'output_col': 'clean_text'
    },
    {
        'id': 'username',
        'name': 'Username Cleaning',
        'icon': 'fa-user-edit',
        'color': '#6366f1',
        'description': 'Clean and normalize usernames. Handles Arabizi, removes symbols, and standardizes formats.',
        'details': [
            'Fill null usernames with "Anonymous"',
            'Convert Arabizi (Mo7amed → Mohamed)',
            'Remove trailing numbers (Hasan855 → Hasan)',
            'Remove punctuation and symbols',
            'Filter names with < 3 letters'
        ],
        'input_col': 'User Name',
        'output_col': 'clean_name'
    },
    {
        'id': 'language',
        'name': 'Language Detection',
        'icon': 'fa-language',
        'color': '#8b5cf6',
        'description': 'Detect the language of each review (Arabic, English, Mixed, or Unknown).',
        'details': [
            'Uses langid library for detection',
            'Checks for Arabic script (Unicode)',
            'Checks for Latin script',
            'Classifies as Arabic, English, Mixed, or Unknown'
        ],
        'input_col': 'Review Text',
        'output_col': 'language'
    },
    {
        'id': 'device',
        'name': 'Device Type Mapping',
        'icon': 'fa-mobile-alt',
        'color': '#06b6d4',
        'description': 'Map platform to device type (iOS, Android, Other).',
        'details': [
            'App Store → iOS',
            'Google Play → Android',
            'Other platforms → Other'
        ],
        'input_col': 'Platform',
        'output_col': 'Device Type'
    },
    {
        'id': 'app_name',
        'name': 'App Name Normalization',
        'icon': 'fa-mobile-screen',
        'color': '#10b981',
        'description': 'Standardize application names to merge duplicates and variations.',
        'details': [
            'نسك → Nusuk نسك',
            'توكلنا → tawakkalna',
            'حافلات مكه → حافلات مكة',
            'Merge spelling variations'
        ],
        'input_col': 'Application Name',
        'output_col': 'Application Name'
    },
    {
        'id': 'service',
        'name': 'Service Classification',
        'icon': 'fa-tags',
        'color': '#f59e0b',
        'description': 'Classify apps into service categories.',
        'details': [
            'Health: صحتي, أسعفني',
            'Reservation: حافلات مكة, قطار الحرمين',
            'Government: توكلنا, نسك',
            'Religious: مكتشف القبله, مصحف الحرمين',
            'Others: Remaining apps'
        ],
        'input_col': 'Application Name',
        'output_col': 'Service_Type'
    },
    {
        'id': 'text_features',
        'name': 'Text Feature Extraction',
        'icon': 'fa-chart-bar',
        'color': '#ec4899',
        'description': 'Extract quantitative text metrics for analysis and modeling.',
        'details': [
            'Character, word, and sentence counts',
            'Arabic/Latin character ratio',
            'Lexical diversity (unique words / total)',
            'Average word length',
            'Punctuation and digit counts',
            'Exclamation and question mark counts'
        ],
        'input_col': 'Review Text',
        'output_col': 'text_word_count'
    },
    {
        'id': 'period',
        'name': 'Period Tagging',
        'icon': 'fa-calendar-alt',
        'color': '#ef4444',
        'description': 'Tag reviews with Islamic calendar events and Saudi school holidays.',
        'details': [
            'Hajj Season (1-15 Dhul Hijjah)',
            'Eid al-Adha (10-13 Dhul Hijjah)',
            'Eid al-Fitr (1-3 Shawwal)',
            'Ramadan (Month 9)',
            'School Summer (KSA dates)',
            'Regular (other periods)'
        ],
        'input_col': 'Review Date',
        'output_col': 'period'
    },
    {
        'id': 'sentiment',
        'name': 'Sentiment Analysis',
        'icon': 'fa-face-smile',
        'color': '#8b5cf6',
        'description': 'Analyze sentiment using deep learning (AraBERT/CAMeL-BERT) for Arabic text.',
        'details': [
            'Uses pre-trained Arabic BERT models',
            'Classifies as Positive, Neutral, or Negative',
            'Provides sentiment score (-1 to 1)',
            'Includes confidence level (0 to 1)',
            'Fallback to lexicon-based if GPU unavailable'
        ],
        'input_col': 'Review Text',
        'output_col': 'sentiment'
    }
]


@app.route('/')
def index():
    """Main page."""
    return render_template('index.html', steps=PIPELINE_STEPS)


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filename = secure_filename(file.filename)
    filepath = app.config['UPLOAD_FOLDER'] / filename
    file.save(filepath)

    try:
        # Load data
        if filename.endswith('.xlsx') or filename.endswith('.xls'):
            df = pd.read_excel(filepath)
        elif filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            return jsonify({'error': 'Unsupported file format. Use .xlsx, .xls, or .csv'}), 400

        # Store in memory with session ID
        session_id = datetime.now().strftime('%Y%m%d%H%M%S')
        data_store[session_id] = {
            'original': df.copy(),
            'current': df.copy(),
            'applied_steps': [],
            'filename': filename
        }

        return jsonify({
            'success': True,
            'session_id': session_id,
            'rows': len(df),
            'columns': list(df.columns),
            'sample': get_sample_data(df),
            'filename': filename
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def safe_str(value):
    """Safely convert any value to string for JSON."""
    if pd.isna(value):
        return ''
    try:
        return str(value)
    except:
        return ''


@app.route('/preview/<session_id>/<step_id>', methods=['GET'])
def preview_step(session_id, step_id):
    """Preview a preprocessing step without applying it."""
    if session_id not in data_store:
        return jsonify({'error': 'Session not found'}), 404

    try:
        df = data_store[session_id]['current'].copy()

        # Get step info
        step_info = next((s for s in PIPELINE_STEPS if s['id'] == step_id), None)
        if not step_info:
            return jsonify({'error': 'Step not found'}), 404

        input_col = step_info['input_col']
        output_col = step_info['output_col']

        # Check if input column exists
        if input_col not in df.columns:
            return jsonify({'error': f'Column "{input_col}" not found in data'}), 400

        before_stats = get_column_stats(df, input_col)

        # Apply transformation
        transformed_df = apply_step(df.copy(), step_id)
        after_stats = get_column_stats(transformed_df, output_col)

        # Get comparison sample - use random sample for variety
        sample_size = min(10, len(df))
        sample_indices = list(range(sample_size))

        comparison = []
        for i in sample_indices:
            before_val = safe_str(df[input_col].iloc[i])
            after_val = safe_str(transformed_df[output_col].iloc[i]) if output_col in transformed_df.columns else 'N/A'
            comparison.append({
                'before': before_val,
                'after': after_val
            })

        return jsonify(clean_for_json({
            'success': True,
            'before_stats': before_stats,
            'after_stats': after_stats,
            'comparison': comparison,
            'input_col': input_col,
            'output_col': output_col
        }))

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/apply/<session_id>/<step_id>', methods=['POST'])
def apply_step_route(session_id, step_id):
    """Apply a preprocessing step."""
    if session_id not in data_store:
        return jsonify({'error': 'Session not found'}), 404

    try:
        df = data_store[session_id]['current'].copy()
        df = apply_step(df, step_id)

        data_store[session_id]['current'] = df
        if step_id not in data_store[session_id]['applied_steps']:
            data_store[session_id]['applied_steps'].append(step_id)

        return jsonify(clean_for_json({
            'success': True,
            'applied_steps': data_store[session_id]['applied_steps'],
            'columns': list(df.columns),
            'sample': get_sample_data(df)
        }))

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def apply_step(df, step_id):
    """Apply a single preprocessing step."""
    if step_id == 'text_clean':
        cleaner = TextCleaner(verbose=False)
        df = cleaner.transform(df)

    elif step_id == 'username':
        processor = UsernamePreprocessor()
        df = processor.transform(df)

    elif step_id == 'language':
        detector = LanguageDetector()
        df = detector.transform(df)

    elif step_id == 'device':
        mapper = DeviceTypeMapper()
        df = mapper.transform(df)

    elif step_id == 'app_name':
        normalizer = AppNameNormalizer()
        df = normalizer.transform(df)

    elif step_id == 'service':
        classifier = ServiceClassifier()
        df = classifier.transform(df)

    elif step_id == 'text_features':
        extractor = TextFeatureExtractor(verbose=False)
        df = extractor.transform(df)

    elif step_id == 'period':
        tagger = PeriodTagger()
        df = tagger.transform(df)
        df = tagger.add_quarter_period(df)

    elif step_id == 'sentiment':
        # Use deep learning if available, otherwise fallback to lexicon
        if TRANSFORMERS_AVAILABLE:
            try:
                analyzer = SentimentAnalyzer(verbose=False)
                df = analyzer.transform(df)
            except Exception:
                # Fallback to simple analyzer on error
                analyzer = SimpleSentimentAnalyzer(verbose=False)
                df = analyzer.transform(df)
        else:
            analyzer = SimpleSentimentAnalyzer(verbose=False)
            df = analyzer.transform(df)

    return df


@app.route('/apply-all/<session_id>', methods=['POST'])
def apply_all_steps(session_id):
    """Apply all preprocessing steps."""
    if session_id not in data_store:
        return jsonify({'error': 'Session not found'}), 404

    try:
        df = data_store[session_id]['original'].copy()

        for step in PIPELINE_STEPS:
            df = apply_step(df, step['id'])

        data_store[session_id]['current'] = df
        data_store[session_id]['applied_steps'] = [s['id'] for s in PIPELINE_STEPS]

        return jsonify(clean_for_json({
            'success': True,
            'applied_steps': data_store[session_id]['applied_steps'],
            'columns': list(df.columns),
            'sample': get_sample_data(df)
        }))

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/reset/<session_id>', methods=['POST'])
def reset_data(session_id):
    """Reset to original data."""
    if session_id not in data_store:
        return jsonify({'error': 'Session not found'}), 404

    data_store[session_id]['current'] = data_store[session_id]['original'].copy()
    data_store[session_id]['applied_steps'] = []

    df = data_store[session_id]['current']

    return jsonify({
        'success': True,
        'columns': list(df.columns),
        'sample': get_sample_data(df)
    })


@app.route('/download/<session_id>', methods=['GET'])
def download_file(session_id):
    """Download processed data."""
    if session_id not in data_store:
        return jsonify({'error': 'Session not found'}), 404

    df = data_store[session_id]['current']
    filename = data_store[session_id]['filename']

    output_filename = f"processed_{filename}"
    output_path = app.config['UPLOAD_FOLDER'] / output_filename

    if output_filename.endswith('.csv'):
        df.to_csv(output_path, index=False)
    else:
        df.to_excel(output_path, index=False)

    return send_file(output_path, as_attachment=True, download_name=output_filename)


@app.route('/stats/<session_id>', methods=['GET'])
def get_stats(session_id):
    """Get dataset statistics."""
    if session_id not in data_store:
        return jsonify({'error': 'Session not found'}), 404

    try:
        df = data_store[session_id]['current']
        original = data_store[session_id]['original']

        stats = {
            'total_rows': int(len(df)),
            'total_columns': int(len(df.columns)),
            'original_columns': int(len(original.columns)),
            'new_columns': int(len(df.columns) - len(original.columns)),
            'applied_steps': int(len(data_store[session_id]['applied_steps'])),
            'total_steps': int(len(PIPELINE_STEPS)),
            'columns': list(df.columns)
        }

        # Column-specific stats
        column_stats = {}
        for col in df.columns:
            # Handle columns with unhashable types (lists, dicts)
            try:
                sample = df[col].dropna().head(5)
                if len(sample) > 0 and isinstance(sample.iloc[0], (list, dict)):
                    unique_count = 'N/A (list column)'
                else:
                    unique_count = int(df[col].nunique())
            except:
                unique_count = 'N/A'

            column_stats[col] = {
                'dtype': str(df[col].dtype),
                'missing': int(df[col].isna().sum()),
                'unique': unique_count
            }

        stats['column_details'] = column_stats

        return jsonify(clean_for_json(stats))

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/analytics/<session_id>', methods=['GET'])
def get_analytics(session_id):
    """Get comprehensive analytics and evaluation metrics."""
    if session_id not in data_store:
        return jsonify({'error': 'Session not found'}), 404

    try:
        df = data_store[session_id]['current']

        # Compute all metrics
        analyzer = DatasetAnalyzer(df)
        metrics = analyzer.compute_all_metrics()

        return jsonify(clean_for_json(metrics))

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/analytics/charts/<session_id>', methods=['GET'])
def generate_charts(session_id):
    """Generate visualization charts."""
    if session_id not in data_store:
        return jsonify({'error': 'Session not found'}), 404

    if not MATPLOTLIB_AVAILABLE:
        return jsonify({'error': 'matplotlib not available for chart generation'}), 500

    try:
        df = data_store[session_id]['current']

        # Generate charts
        chart_dir = app.config['UPLOAD_FOLDER'] / 'charts' / session_id
        chart_dir.mkdir(parents=True, exist_ok=True)

        visualizer = DatasetVisualizer(df, str(chart_dir))
        charts = visualizer.generate_all_charts()

        # Convert paths to URLs
        chart_urls = {}
        for name, path in charts.items():
            if path and 'error' not in name:
                chart_urls[name] = f'/charts/{session_id}/{Path(path).name}'

        return jsonify({
            'success': True,
            'charts': chart_urls
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/charts/<session_id>/<filename>')
def serve_chart(session_id, filename):
    """Serve generated chart images."""
    chart_path = app.config['UPLOAD_FOLDER'] / 'charts' / session_id / filename
    if chart_path.exists():
        return send_file(chart_path, mimetype='image/png')
    return jsonify({'error': 'Chart not found'}), 404


@app.route('/analytics/report/<session_id>', methods=['GET'])
def get_report(session_id):
    """Get text summary report."""
    if session_id not in data_store:
        return jsonify({'error': 'Session not found'}), 404

    try:
        df = data_store[session_id]['current']

        analyzer = DatasetAnalyzer(df)
        analyzer.compute_all_metrics()
        report = analyzer.generate_summary_report()

        return jsonify({
            'success': True,
            'report': report
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
