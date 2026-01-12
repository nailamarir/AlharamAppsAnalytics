# AlHaram Analytics: Technical Report

## Preprocessing Pipeline for Saudi Arabian Mobile Application Reviews

**Version:** 0.1.0
**Date:** January 2026
**Authors:** Naila Marir

---

## Executive Summary

AlHaram Analytics is a specialized data preprocessing and analytics system designed to process, clean, and enrich mobile application review data from Saudi Arabian government and religious service applications. The system addresses unique challenges in Arabic text processing, including Arabizi conversion, Islamic calendar event tagging, and multi-language support. The pipeline processes reviews from applications related to Hajj, Umrah, healthcare, transportation, and government services, providing structured data suitable for sentiment analysis and user experience research.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [System Architecture](#2-system-architecture)
3. [Technology Stack](#3-technology-stack)
4. [Core Components](#4-core-components)
5. [Data Processing Pipeline](#5-data-processing-pipeline)
6. [Web Application](#6-web-application)
7. [API Reference](#7-api-reference)
8. [Data Schema](#8-data-schema)
9. [Installation and Deployment](#9-installation-and-deployment)
10. [Performance Considerations](#10-performance-considerations)
11. [Future Enhancements](#11-future-enhancements)

---

## 1. Introduction

### 1.1 Background

The Kingdom of Saudi Arabia has developed numerous mobile applications to support pilgrims, residents, and visitors. These applications span healthcare (Sehhaty, Asaafni), transportation (Makkah Buses, Haramain Train), government services (Tawakkalna, Nusuk), and religious guidance (Quran apps, Qibla finders). User reviews from these applications provide valuable insights into user experience, service quality, and areas for improvement.

### 1.2 Problem Statement

Processing Arabic app reviews presents unique challenges:
- **Arabizi usage**: Users frequently write Arabic words using Latin characters and numerals (e.g., "7abibi" for "حبيبي")
- **Multi-language content**: Reviews contain Arabic, English, or mixed-language text
- **Islamic calendar relevance**: User behavior varies significantly during Hajj, Ramadan, and Eid periods
- **Username diversity**: Usernames contain Arabic, Latin, emojis, and special characters

### 1.3 Objectives

1. Standardize and clean user-generated content
2. Detect and classify review languages
3. Tag reviews with relevant Islamic calendar periods
4. Classify applications by service type
5. Predict user gender from usernames (optional)
6. Provide interactive visualization and processing interface

### 1.4 Target Applications

| Category | Applications |
|----------|-------------|
| **Health Services** | صحتي (Sehhaty), أسعفني (Asaafni) |
| **Transportation** | حافلات مكة (Makkah Buses), قطار الحرمين (HHR Train), تنقل (Tanqul) |
| **Government** | توكلنا (Tawakkalna), نسك (Nusuk), ارشاد (Irshad) |
| **Religious** | مكتشف القبله (Qibla Finder), مصحف الحرمين (Haramain Quran) |

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     AlHaram Analytics System                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  Web Interface│    │  CLI Interface│    │  Python API  │      │
│  │   (Flask)     │    │   (argparse)  │    │   (Module)   │      │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘      │
│         │                   │                   │               │
│         └───────────────────┼───────────────────┘               │
│                             │                                    │
│                    ┌────────▼────────┐                          │
│                    │ PreprocessingPipeline                      │
│                    └────────┬────────┘                          │
│                             │                                    │
│  ┌──────────────────────────┼──────────────────────────┐        │
│  │                          │                          │        │
│  ▼                          ▼                          ▼        │
│ ┌────────────┐    ┌─────────────────┐    ┌─────────────────┐   │
│ │Preprocessing│    │Feature Engineering│   │Gender Prediction│   │
│ ├────────────┤    ├─────────────────┤    ├─────────────────┤   │
│ │• Username  │    │• App Normalizer │    │• HF Classifier  │   │
│ │• Language  │    │• Device Mapper  │    │• Ensemble       │   │
│ └────────────┘    │• Period Tagger  │    └─────────────────┘   │
│                   │• Service Class. │                          │
│                   └─────────────────┘                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Directory Structure

```
AlharamApplication/
├── src/alharam_analytics/          # Core library
│   ├── __init__.py
│   ├── pipeline.py                 # Main orchestrator
│   ├── preprocessing/              # Text preprocessing
│   │   ├── username_cleaner.py
│   │   └── language_detector.py
│   ├── feature_engineering/        # Feature extraction
│   │   ├── app_name_normalizer.py
│   │   ├── device_mapper.py
│   │   ├── period_tagger.py
│   │   └── service_classifier.py
│   ├── gender_prediction/          # ML-based gender prediction
│   │   ├── hf_gender_classifier.py
│   │   └── ensemble_predictor.py
│   └── utils/
│       └── io_utils.py
├── webapp/                         # Flask web application
│   ├── app.py
│   ├── templates/
│   └── static/
├── config/
│   └── pipeline_config.yaml
├── scripts/
│   └── run_preprocessing.py
├── docs/
├── notebooks/
└── data/
    ├── raw/
    └── processed/
```

### 2.3 Design Principles

1. **Modularity**: Each preprocessing step is an independent, reusable component
2. **Scikit-learn Compatibility**: Transformers implement fit/transform interface
3. **Configurability**: YAML-based configuration for all parameters
4. **Extensibility**: Easy to add new preprocessing steps or classifiers
5. **Multi-interface**: Web UI, CLI, and Python API for different use cases

---

## 3. Technology Stack

### 3.1 Core Technologies

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| Language | Python | 3.10+ | Primary development language |
| Data Processing | pandas | ≥2.0.0 | DataFrame operations |
| Excel Support | openpyxl | ≥3.1.0 | Excel file I/O |
| Language Detection | langid | ≥1.1.6 | Text language classification |
| Islamic Calendar | hijri-converter | ≥2.3.0 | Gregorian-Hijri conversion |
| Web Framework | Flask | ≥3.0.0 | REST API and web interface |
| ML Models | transformers | ≥4.30.0 | HuggingFace model inference |
| Deep Learning | PyTorch | ≥2.0.0 | Model backend |

### 3.2 Frontend Technologies

- **HTML5**: Semantic markup
- **CSS3**: Responsive styling with CSS Grid and Flexbox
- **JavaScript (ES6+)**: Async/await, fetch API
- **Font Awesome 6.5.1**: Icon library
- **Google Fonts**: Inter (Latin), Cairo (Arabic)

### 3.3 Development Tools

| Tool | Purpose |
|------|---------|
| pytest | Unit testing |
| black | Code formatting (line length: 88) |
| ruff | Linting (E, F, I, N, W rules) |
| setuptools | Package building |

---

## 4. Core Components

### 4.1 Username Cleaner

**Module:** `src/alharam_analytics/preprocessing/username_cleaner.py`

Handles diverse username formats common in Arabic-speaking user bases.

#### Arabizi Conversion Map

| Character | Arabic Equivalent | Example |
|-----------|-------------------|---------|
| 2 | ء (hamza) | a2mad → ahmad |
| 3 | ع (ain) | 3ali → ali |
| 5 | خ (kha) | 5aled → khaled |
| 6 | ط (ta) | 6ariq → tariq |
| 7 | ح (ha) | a7mad → ahmad |
| 8 | غ (ghain) | 8areeb → ghareeb |
| 9 | ص (sad) | 9aber → saber |

#### Processing Steps

1. Convert Arabizi numerals to Latin equivalents
2. Remove trailing digits (e.g., "Hassan855" → "Hassan")
3. Strip punctuation, symbols, and emojis
4. Filter names with fewer than 3 letters → "Anonymous"
5. Handle null/None values gracefully

```python
# Example usage
from alharam_analytics.preprocessing import UsernamePreprocessor

cleaner = UsernamePreprocessor()
df = cleaner.transform(df)  # Adds "clean_name" column
```

### 4.2 Language Detector

**Module:** `src/alharam_analytics/preprocessing/language_detector.py`

Multi-strategy language detection for Arabic/English content.

#### Detection Algorithm

```
1. Use langid library for primary classification
2. Check for Arabic Unicode range (U+0600-U+06FF)
3. Check for Latin characters (A-Za-z)
4. Classification:
   - Primarily Arabic characters → "Arabic"
   - Primarily Latin characters → "English"
   - Both present significantly → "Mixed"
   - Neither detected → "Unknown"
```

#### Output Categories

| Category | Description |
|----------|-------------|
| Arabic | Predominantly Arabic script |
| English | Predominantly Latin script |
| Mixed | Significant presence of both scripts |
| Unknown | Unable to determine (emojis only, symbols, etc.) |

### 4.3 Period Tagger

**Module:** `src/alharam_analytics/feature_engineering/period_tagger.py`

Tags reviews based on Islamic calendar events and Saudi academic calendar.

#### Islamic Period Definitions

| Period | Hijri Date | Description |
|--------|------------|-------------|
| Hajj Season | 1-15 Dhul Hijjah | Peak pilgrimage period |
| Eid al-Adha | 10-13 Dhul Hijjah | Festival of Sacrifice |
| Eid al-Fitr | 1-3 Shawwal | End of Ramadan |
| Ramadan | Full month 9 | Fasting month |
| School Summer | Ministry dates | Saudi academic break |
| Regular | Other dates | Normal period |

#### Implementation

```python
from hijri_converter import Hijri, Gregorian

def tag_period(gregorian_date):
    hijri = Gregorian(date.year, date.month, date.day).to_hijri()

    if hijri.month == 12 and 1 <= hijri.day <= 15:
        if 10 <= hijri.day <= 13:
            return "Eid al-Adha"
        return "Hajj Season"
    elif hijri.month == 10 and 1 <= hijri.day <= 3:
        return "Eid al-Fitr"
    elif hijri.month == 9:
        return "Ramadan"
    # ... check school summer dates
    return "Regular"
```

### 4.4 Service Classifier

**Module:** `src/alharam_analytics/feature_engineering/service_classifier.py`

Categorizes applications into service types.

#### Classification Mapping

```python
SERVICE_MAPPING = {
    "Health Services": [
        "صحتي", "Sehhaty", "أسعفني", "Asaafni"
    ],
    "Reservation/Transport": [
        "حافلات مكة", "Makkah Buses", "قطار الحرمين", "HHR Train",
        "تنقل", "Tanqul", "تروية", "Trwayyah"
    ],
    "Government Services": [
        "توكلنا", "Tawakkalna", "نسك", "Nusuk", "ارشاد", "Irshad"
    ],
    "Religious": [
        "مكتشف القبله", "Qibla Finder", "مصحف الحرمين", "فاذكروني"
    ],
    "Others": []  # Default category
}
```

### 4.5 Gender Prediction (Optional)

**Module:** `src/alharam_analytics/gender_prediction/`

Ensemble-based gender prediction using HuggingFace transformers.

#### Models Used

1. **imranali291/genderize**: General name-based classifier
2. **padmajabfrl/Gender-Classification**: Alternative classifier

#### Ensemble Decision Logic

```python
def predict_ensemble(name):
    pred1, conf1 = model1.predict(name)
    pred2, conf2 = model2.predict(name)

    if pred1 == pred2:
        return pred1  # Agreement
    elif conf1 >= 0.80:
        return pred1  # High confidence model 1
    elif conf2 >= 0.80:
        return pred2  # High confidence model 2
    else:
        return "unknown"  # Disagreement, low confidence
```

#### Output Columns

| Column | Description |
|--------|-------------|
| pred_gender_1 | Model 1 prediction |
| pred_score_1 | Model 1 confidence (0-1) |
| pred_gender_2 | Model 2 prediction |
| pred_score_2 | Model 2 confidence (0-1) |
| gender_final | Ensemble decision |

---

## 5. Data Processing Pipeline

### 5.1 Pipeline Overview

The `PreprocessingPipeline` class orchestrates all preprocessing steps in a configurable sequence.

```python
from alharam_analytics.pipeline import PreprocessingPipeline

# Initialize
pipeline = PreprocessingPipeline(
    include_gender_prediction=False,
    verbose=True
)

# Run full pipeline
df = pipeline.run("data/raw/reviews.xlsx")

# Run specific steps
df = pipeline.run(
    "data/raw/reviews.xlsx",
    steps=["username", "language", "period"]
)

# Save results
pipeline.save(df, "data/processed/reviews_cleaned.xlsx")
```

### 5.2 Pipeline Steps

| Step | Component | Input Column | Output Column |
|------|-----------|--------------|---------------|
| 1 | UsernamePreprocessor | User Name | clean_name |
| 2 | LanguageDetector | Review Text | language |
| 3 | DeviceTypeMapper | Platform | Device Type |
| 4 | AppNameNormalizer | Application Name | Application Name (normalized) |
| 5 | ServiceClassifier | Application Name | Service_Type |
| 6 | PeriodTagger | Review Date | period, App_Version_Period |
| 7* | GenderEnsemblePredictor | clean_name | gender_final, pred_* |

*Optional step

### 5.3 Pipeline Execution Flow

```
Input Data (Excel/CSV)
         │
         ▼
┌─────────────────────┐
│ 1. Username Cleaning │
│    - Arabizi conv.   │
│    - Symbol removal  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ 2. Language Detection│
│    - langid          │
│    - Script analysis │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ 3. Device Mapping    │
│    - Platform → Type │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ 4. App Normalization │
│    - Spelling fixes  │
│    - Name unification│
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ 5. Service Classify  │
│    - Category assign │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ 6. Period Tagging    │
│    - Hijri convert   │
│    - Event matching  │
└──────────┬──────────┘
           │
           ▼ (optional)
┌─────────────────────┐
│ 7. Gender Prediction │
│    - HF models       │
│    - Ensemble vote   │
└──────────┬──────────┘
           │
           ▼
    Output Data
```

---

## 6. Web Application

### 6.1 Overview

The Flask-based web application provides an interactive interface for data preprocessing with real-time preview and step-by-step execution.

**Entry Point:** `run_webapp.py`
**Default URL:** http://localhost:5000

### 6.2 User Interface Components

#### File Upload Section
- Drag-and-drop or click to browse
- Supports .xlsx, .xls, .csv formats
- Maximum file size: 50MB
- Upload progress indication

#### Pipeline Visualization
Six processing cards with:
- Step icon and name
- Brief description
- Preview button (shows before/after)
- Apply button (executes step)
- Status indicator (pending/applied)

#### Data Preview
- Sample rows display (first 10 rows)
- Column headers with data types
- Scrollable table view
- Column statistics (unique, missing, top values)

#### Action Buttons
- **Apply All**: Execute full pipeline
- **Reset**: Revert to original data
- **Download**: Export processed file
- **Change File**: Upload different file

### 6.3 Session Management

```python
# In-memory session storage
sessions = {
    "20260103143052": {
        "original_df": DataFrame,    # Original uploaded data
        "current_df": DataFrame,     # Current state after transformations
        "applied_steps": ["username", "language"],  # Applied steps
        "filename": "reviews.xlsx"   # Original filename
    }
}
```

**Note:** Production deployments should use Redis or database-backed sessions.

---

## 7. API Reference

### 7.1 REST Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main web interface |
| `/upload` | POST | Upload data file |
| `/preview/<session_id>/<step_id>` | GET | Preview transformation |
| `/apply/<session_id>/<step_id>` | POST | Apply single step |
| `/apply-all/<session_id>` | POST | Apply all steps |
| `/reset/<session_id>` | POST | Reset to original |
| `/download/<session_id>` | GET | Download processed file |
| `/stats/<session_id>` | GET | Get dataset statistics |

### 7.2 Response Formats

#### Upload Response
```json
{
  "success": true,
  "session_id": "20260103143052",
  "filename": "reviews.xlsx",
  "rows": 15420,
  "columns": ["Review ID", "Review Date", "Review Text", ...],
  "sample": [...]
}
```

#### Preview Response
```json
{
  "success": true,
  "step": "username",
  "input_column": "User Name",
  "output_column": "clean_name",
  "sample_before": ["Mo7amed123", "فاطمة_2020", ...],
  "sample_after": ["Mohamed", "فاطمة", ...],
  "stats": {
    "unique_before": 8542,
    "unique_after": 7831,
    "anonymous_count": 245
  }
}
```

### 7.3 CLI Interface

```bash
# Basic usage
python scripts/run_preprocessing.py \
    -i data/raw/reviews.xlsx \
    -o data/processed/reviews_cleaned.xlsx

# With gender prediction
python scripts/run_preprocessing.py \
    -i data/raw/reviews.xlsx \
    -o data/processed/reviews_cleaned.xlsx \
    --gender

# Quiet mode (no progress output)
python scripts/run_preprocessing.py \
    -i data/raw/reviews.xlsx \
    -o data/processed/reviews_cleaned.xlsx \
    -q

# Specific steps only
python scripts/run_preprocessing.py \
    -i data/raw/reviews.xlsx \
    -o data/processed/reviews_cleaned.xlsx \
    --steps username language period
```

---

## 8. Data Schema

### 8.1 Input Requirements

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| Review ID | integer | Yes | Unique identifier |
| Review Date | datetime | Yes | Review submission date |
| Review Text | string | Yes | Review content |
| Rating | integer | No | 1-5 star rating |
| User Name | string | Yes | Reviewer username |
| Platform | string | Yes | App store (App Store/Google Play) |
| Application Name | string | Yes | Application name |

### 8.2 Output Schema

After full pipeline execution:

| Column | Type | Source | Description |
|--------|------|--------|-------------|
| Review ID | int | Original | Unique identifier |
| Review Date | datetime | Original | Submission date |
| Review Text | string | Original | Review content |
| Rating | int | Original | Star rating |
| User Name | string | Original | Original username |
| **clean_name** | string | Step 1 | Cleaned username |
| Platform | string | Original | App store |
| **Device Type** | string | Step 3 | iOS/Android/Other |
| Application Name | string | Step 4 | Normalized app name |
| **Service_Type** | string | Step 5 | Service category |
| **language** | string | Step 2 | Arabic/English/Mixed/Unknown |
| **period** | string | Step 6 | Islamic calendar period |
| **App_Version_Period** | string | Step 6 | Quarter (2023Q1, etc.) |
| **gender_final*** | string | Step 7 | Male/Female/unknown |
| **pred_gender_1*** | string | Step 7 | Model 1 prediction |
| **pred_score_1*** | float | Step 7 | Model 1 confidence |
| **pred_gender_2*** | string | Step 7 | Model 2 prediction |
| **pred_score_2*** | float | Step 7 | Model 2 confidence |

*Only if gender prediction enabled

---

## 9. Installation and Deployment

### 9.1 Requirements

- Python 3.10 or higher
- pip package manager
- 4GB+ RAM (8GB+ recommended for gender prediction)

### 9.2 Installation Steps

```bash
# Clone repository
git clone https://github.com/username/AlharamApplication.git
cd AlharamApplication

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Optional: Install gender prediction dependencies
pip install transformers torch

# Install package in development mode
pip install -e .
```

### 9.3 Running the Application

#### Web Interface
```bash
python run_webapp.py
# Opens browser at http://localhost:5000
```

#### Command Line
```bash
python scripts/run_preprocessing.py -i input.xlsx -o output.xlsx
```

#### Python API
```python
from alharam_analytics.pipeline import PreprocessingPipeline

pipeline = PreprocessingPipeline()
df = pipeline.run("input.xlsx")
pipeline.save(df, "output.xlsx")
```

### 9.4 Configuration

Edit `config/pipeline_config.yaml`:

```yaml
data:
  raw_dir: "data/raw"
  processed_dir: "data/processed"

preprocessing:
  username:
    min_letters: 3
    column_name: "User Name"
    output_column: "clean_name"

gender_prediction:
  enabled: false
  confidence_threshold: 0.60
  high_confidence_threshold: 0.80

logging:
  verbose: true
```

---

## 10. Performance Considerations

### 10.1 Processing Times

| Step | Time per 1000 rows | Notes |
|------|-------------------|-------|
| Username Cleaning | ~0.5 seconds | String operations |
| Language Detection | ~2-3 seconds | langid inference |
| Device Mapping | ~0.1 seconds | Dictionary lookup |
| App Normalization | ~0.2 seconds | String matching |
| Service Classification | ~0.2 seconds | Dictionary lookup |
| Period Tagging | ~1-2 seconds | Hijri conversion |
| Gender Prediction | ~30-60 seconds | Transformer inference |

### 10.2 Memory Usage

| Dataset Size | Without Gender | With Gender |
|--------------|----------------|-------------|
| 10,000 rows | ~200 MB | ~2 GB |
| 50,000 rows | ~500 MB | ~3 GB |
| 100,000 rows | ~1 GB | ~4 GB |

### 10.3 Optimization Recommendations

1. **Batch Processing**: Process large files in chunks
2. **Disable Gender Prediction**: Skip if not needed (saves 80% time)
3. **Selective Steps**: Run only required preprocessing steps
4. **Caching**: Cache HuggingFace models locally

---

## 11. Future Enhancements

### 11.1 Planned Features

1. **Sentiment Analysis Integration**: Add Arabic sentiment classification
2. **Topic Modeling**: Automatic topic extraction from reviews
3. **Database Backend**: PostgreSQL for persistent storage
4. **User Authentication**: Multi-user support with sessions
5. **Batch Processing API**: Async processing for large datasets
6. **Export Formats**: Add PDF report generation

### 11.2 Technical Improvements

1. **Redis Sessions**: Replace in-memory session storage
2. **Celery Workers**: Background task processing
3. **Docker Deployment**: Containerized deployment
4. **API Rate Limiting**: Production-ready API protection
5. **Logging**: Structured logging with ELK stack integration

---

## Appendix A: Arabizi Character Mapping

| Number | Arabic Letter | Name | Example |
|--------|---------------|------|---------|
| 2 | ء / أ | Hamza/Alif | sa2al → سأل |
| 3 | ع | Ain | 3arab → عرب |
| 5 | خ | Kha | 5air → خير |
| 6 | ط | Ta | 6areq → طريق |
| 7 | ح | Ha | 7ubb → حب |
| 8 | غ | Ghain | 8areeb → غريب |
| 9 | ص | Sad | 9abr → صبر |

## Appendix B: Islamic Calendar Reference

| Hijri Month | Number | Notable Events |
|-------------|--------|----------------|
| Muharram | 1 | Islamic New Year |
| Safar | 2 | - |
| Rabi' al-Awwal | 3 | Mawlid |
| Rabi' al-Thani | 4 | - |
| Jumada al-Awwal | 5 | - |
| Jumada al-Thani | 6 | - |
| Rajab | 7 | Isra and Mi'raj |
| Sha'ban | 8 | - |
| Ramadan | 9 | Fasting Month |
| Shawwal | 10 | Eid al-Fitr |
| Dhul Qa'dah | 11 | - |
| Dhul Hijjah | 12 | Hajj, Eid al-Adha |

---

## References

1. Flask Documentation: https://flask.palletsprojects.com/
2. HuggingFace Transformers: https://huggingface.co/docs/transformers/
3. Hijri Converter: https://hijri-converter.readthedocs.io/
4. langid.py: https://github.com/saffsd/langid.py

---

*Document generated: January 2026*
*AlHaram Analytics v0.1.0*
