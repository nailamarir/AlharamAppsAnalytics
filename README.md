# AlHaram Analytics

A comprehensive data preprocessing and analytics system for Saudi Arabian mobile application reviews, specializing in Arabic NLP, Islamic calendar integration, and multi-language processing.

## 🎯 Overview

AlHaram Analytics processes and analyzes user reviews from Saudi Arabian government and religious service applications including healthcare (Sehhaty), transportation (Makkah Buses, Haramain Train), and government services (Tawakkalna, Nusuk). The system addresses unique challenges in Arabic text processing, including Arabizi conversion, Islamic calendar event tagging, and multi-language support.

## 🌟 Key Features

### 📊 Data Processing Pipeline
- **Arabizi Conversion**: Transliterates Latin-based Arabic (e.g., "7abibi" → "حبيبي")
- **Multi-language Detection**: Identifies Arabic, English, and mixed-language text
- **Username Cleaning**: Standardizes usernames with Arabic/Latin/emoji content
- **Text Normalization**: Handles diacritics, special characters, and formatting

### 🕌 Islamic Context Integration
- **Hijri Calendar Integration**: Automatic Islamic date conversion
- **Temporal Tagging**: Identifies reviews during Hajj, Ramadan, Eid, or regular periods
- **Cultural Feature Extraction**: Saudi-specific contextual features

### 🤖 Analytics & ML
- **Sentiment Analysis**: Arabic sentiment classification (59.2% accuracy baseline)
- **Gender Prediction**: Optional username-based gender inference
- **Service Categorization**: Auto-classification by app type (Health, Transport, Government, Religious)

### 📈 Visualization & Reporting
- **11+ Interactive Charts**: Distribution, temporal, sentiment analysis
- **Comprehensive Reports**: Technical documentation and research papers
- **Web Interface**: Flask-based interactive dashboard

## 📁 Project Structure

```
AlharamApplication/
├── src/
│   └── alharam_analytics/
│       ├── preprocessing/         # Data cleaning and normalization
│       ├── feature_engineering/   # App classification, period tagging
│       ├── sentiment/             # Sentiment analysis models
│       ├── gender_prediction/     # Optional gender inference
│       ├── analytics/             # Statistical analysis
│       └── utils/                 # Helper functions
├── webapp/                        # Flask web interface
├── taxonomy-diagram/              # Interactive framework visualization (React)
├── docs/
│   ├── TECHNICAL_REPORT.pdf       # 47-page technical documentation
│   ├── RESEARCH_PAPER.pdf         # 28-page academic paper
│   ├── PIPELINE.md                # Pipeline documentation
│   └── DATA_QUALITY_GUIDE.md      # Data quality guidelines
├── output/
│   ├── charts/                    # Generated visualizations
│   └── processed_dataset.xlsx     # Processed data (57K+ reviews)
├── data/                          # Input data directory
├── scripts/                       # Utility scripts
├── generate_visualizations.py     # Automated chart generation
├── evaluate_sentiment.py          # Model evaluation script
├── run_webapp.py                  # Web app launcher
└── requirements.txt               # Python dependencies
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- pip or conda

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/AlharamApplication.git
cd AlharamApplication

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline

```bash
# Process dataset
python -m src.alharam_analytics.pipeline --input dataset.xlsx --output output/

# Generate visualizations
python generate_visualizations.py

# Evaluate sentiment analysis
python evaluate_sentiment.py
```

### Running the Web Interface

```bash
python run_webapp.py
# Open browser to http://localhost:5000
```

## 📊 Sample Results

**Dataset**: 57,717 mobile app reviews processed

**Sentiment Analysis Performance**:
- Overall Accuracy: 59.2%
- Positive Class: 75.1% F1-score (41,683 samples)
- Negative Class: 14.4% F1-score (13,430 samples)
- Neutral Class: 6.4% F1-score (2,604 samples)

**Target Applications** (15+ apps):
- 🏥 Healthcare: صحتي (Sehhaty), أسعفني (Asaafni)
- 🚌 Transportation: حافلات مكة (Makkah Buses), قطار الحرمين (HHR Train)
- 🏛️ Government: توكلنا (Tawakkalna), نسك (Nusuk)
- 🕌 Religious: مصحف الحرمين (Quran app), مكتشف القبله (Qibla Finder)

## 🎨 Interactive Taxonomy Visualization

The `taxonomy-diagram/` directory contains a React-based interactive visualization of the framework:

```bash
cd taxonomy-diagram
npm install
npm run dev
# Open browser to http://localhost:5173
```

**Framework Levels**:
1. 🏛️ Framework Overview
2. ✍️ Level 1: Lexical Quality
3. 🌐 Level 2: Linguistic Quality
4. 🕌 Level 3: Islamic Context
5. 📅 Temporal-Cultural Injection
6. 🔗 Cross-Linguistic Unification

## 🔧 Core Components

### 1. Preprocessing Module

```python
from alharam_analytics.preprocessing import DataCleaner

cleaner = DataCleaner()
cleaned_text = cleaner.clean_review(text)
language = cleaner.detect_language(text)
```

### 2. Feature Engineering

```python
from alharam_analytics.feature_engineering import PeriodTagger

tagger = PeriodTagger()
period = tagger.tag_period(date)  # Returns: Hajj/Ramadan/Eid/Regular
```

### 3. Sentiment Analysis

```python
from alharam_analytics.sentiment import SentimentAnalyzer

analyzer = SentimentAnalyzer()
sentiment = analyzer.predict(text)  # Returns: positive/neutral/negative
```

## 📖 Documentation

- **[Technical Report](docs/TECHNICAL_REPORT.pdf)**: Complete system architecture and implementation
- **[Research Paper](docs/RESEARCH_PAPER.pdf)**: Academic presentation of methodology
- **[Pipeline Guide](docs/PIPELINE.md)**: Step-by-step processing workflow
- **[Data Quality Guide](docs/DATA_QUALITY_GUIDE.md)**: Quality assurance guidelines

## 🧪 Testing

```bash
# Run tests (if available)
pytest tests/

# Run specific module test
python -m pytest tests/test_preprocessing.py
```

## 📈 Visualization Examples

The system generates 11+ chart types:

1. **Distribution Analysis**:
   - Reviews per application
   - Language distribution
   - Device distribution

2. **Temporal Analysis**:
   - Reviews over time
   - Islamic period comparison
   - Seasonal patterns

3. **Sentiment Analysis**:
   - Sentiment by app
   - Sentiment by period
   - Correlation with ratings

4. **Evaluation Metrics**:
   - Confusion matrices
   - Performance heatmaps

## 🎓 Research Context

This project is part of academic research on:
- Arabic Natural Language Processing
- Cross-linguistic data processing
- Islamic calendar integration in analytics
- Mobile app review analysis
- Sentiment analysis for Arabic text

## 🤝 Key Contributions

1. **Arabizi Processing**: Novel approach to Latin-based Arabic text
2. **Islamic Calendar Integration**: First framework to integrate Hijri dates in app analytics
3. **Multi-language Taxonomy**: Unified framework for Arabic-English-Arabizi processing
4. **Cultural Context Awareness**: Saudi-specific feature extraction

## 📊 Technologies Used

**Core**:
- Python 3.10+
- pandas, numpy
- scikit-learn

**NLP**:
- langid (language detection)
- transformers (sentiment analysis)
- Arabic text processing libraries

**Visualization**:
- matplotlib, seaborn
- React + Vite (interactive diagrams)

**Web**:
- Flask (web interface)
- HTML/CSS/JavaScript

**Calendar**:
- hijri-converter (Islamic calendar)

## 🐛 Known Issues & Limitations

1. **Sentiment Analysis**: Struggles with neutral class (6.4% F1) due to class imbalance
2. **Arabizi Conversion**: Limited to common patterns
3. **Language Detection**: May misclassify very short texts
4. **Performance**: Large datasets (>100K reviews) require significant memory

## 🔮 Future Enhancements

- [ ] Improve neutral sentiment detection
- [ ] Expand Arabizi conversion dictionary
- [ ] Add multi-label classification
- [ ] Implement real-time processing API
- [ ] Add support for more dialects
- [ ] Integrate advanced Arabic models (CAMeL, AraBERT)

## 👤 Author

**Naila Marir**
- Email: nailamarir@email.com

## 📄 License

This project is part of academic research. For usage permissions and collaboration opportunities, please contact the author.

## 🙏 Acknowledgments

- Saudi Arabian app developers for creating valuable services
- Open-source Arabic NLP community
- Islamic calendar conversion libraries

## 📝 Citation

If you use this framework in your research, please cite:

```bibtex
@software{alharam_analytics_2026,
  author = {Marir, Naila},
  title = {AlHaram Analytics: A Multilingual Data Quality Framework for Islamic Service Analytics},
  year = {2026},
  url = {https://github.com/YOUR_USERNAME/AlharamApplication}
}
```

## 📞 Contact & Support

For questions, issues, or collaboration:
- Email: nailamarir@email.com
- Issues: GitHub Issues tab
- Documentation: See `docs/` directory

---

**Note**: This system handles sensitive user data. Ensure compliance with data privacy regulations (GDPR, PDPL) when processing real user reviews.
