# AlHaram Analytics Pipeline Documentation

## Overview

This pipeline processes app reviews from AlHaram-related mobile applications (Hajj, Umrah, government services) to create a clean, enriched dataset for analytics.

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RAW DATA INPUT                                     │
│                        (Excel/CSV reviews)                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        1. USERNAME PREPROCESSING                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ • Fill null usernames with "Anonymous"                              │    │
│  │ • Remove punctuation, symbols, emojis                               │    │
│  │ • Convert Arabizi numbers to letters (Mo7amed → Mohamed)            │    │
│  │ • Remove trailing digits (Hasan855 → Hasan)                         │    │
│  │ • Filter names with < 3 letters → "Anonymous"                       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│  Input:  "User Name" column                                                  │
│  Output: "clean_name" column                                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        2. LANGUAGE DETECTION                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ • Detect primary language using langid library                      │    │
│  │ • Check for Arabic script (Unicode range \u0600-\u06FF)             │    │
│  │ • Check for Latin script (A-Za-z)                                   │    │
│  │ • Classify as: Arabic, English, Mixed, or Unknown                   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│  Input:  "Review Text" column                                                │
│  Output: "language" column                                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        3. DEVICE TYPE MAPPING                                │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ • Map "App Store" → iOS                                             │    │
│  │ • Map "Google Play" → Android                                       │    │
│  │ • Others → Other                                                    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│  Input:  "Platform" column                                                   │
│  Output: "Device Type" column                                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     4. APP NAME NORMALIZATION                                │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ • Standardize spelling variations:                                  │    │
│  │   - "نسك" → "Nusuk نسك"                                             │    │
│  │   - "توكلنا" → "tawakkalna"                                         │    │
│  │   - "حافلات مكه" → "حافلات مكة"                                     │    │
│  │ • Merge duplicate app entries                                       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│  Input:  "Application Name" column                                           │
│  Output: Normalized "Application Name" column                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     5. SERVICE TYPE CLASSIFICATION                           │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ Categories:                                                         │    │
│  │ • Health: صحتي, أسعفني                                              │    │
│  │ • Reservation: حافلات مكة, قطار الحرمين, تنقل                       │    │
│  │ • Government Services: توكلنا, نسك, ارشاد                           │    │
│  │ • Religious: مكتشف القبله, مصحف الحرمين                             │    │
│  │ • Others: Remaining apps                                            │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│  Input:  "Application Name" column                                           │
│  Output: "Service_Type" column                                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        6. PERIOD TAGGING                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ Islamic Calendar Events (using Hijri conversion):                   │    │
│  │ • Hajj Season: 1-15 Dhul Hijjah                                     │    │
│  │ • Eid al-Adha: 10-13 Dhul Hijjah                                    │    │
│  │ • Eid al-Fitr: 1-3 Shawwal                                          │    │
│  │ • Ramadan: Month 9                                                  │    │
│  │                                                                     │    │
│  │ Saudi Calendar Events:                                              │    │
│  │ • School Summer: KSA Ministry of Education dates (2012-2025)        │    │
│  │ • Regular: All other periods                                        │    │
│  │                                                                     │    │
│  │ Also adds quarterly period (Q1 2023, Q2 2023, etc.)                 │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│  Input:  "Review Date" column                                                │
│  Output: "period" + "App_Version_Period" columns                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   7. GENDER PREDICTION (Optional)                            │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ Uses ensemble of two HuggingFace models:                            │    │
│  │ • Model 1: imranali291/genderize                                    │    │
│  │ • Model 2: padmajabfrl/Gender-Classification                        │    │
│  │                                                                     │    │
│  │ Agreement Logic:                                                    │    │
│  │ • If both models agree → use that gender                            │    │
│  │ • If one has confidence ≥ 0.80 → trust that model                   │    │
│  │ • Otherwise → "unknown"                                             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│  Input:  "clean_name" column                                                 │
│  Output: "gender_final", "pred_gender_1", "pred_gender_2" columns            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PROCESSED OUTPUT                                     │
│                    (Clean, enriched dataset)                                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Output Columns

After running the full pipeline, your dataset will have these additional columns:

| Column | Description | Example Values |
|--------|-------------|----------------|
| `clean_name` | Cleaned username | "Mohamed", "Sara", "Anonymous" |
| `language` | Review language | "Arabic", "English", "Mixed", "Unknown" |
| `Device Type` | Device platform | "iOS", "Android", "Other" |
| `Service_Type` | App category | "Health", "Reservation", "Government Services", "Religious", "Others" |
| `period` | Special period tag | "Hajj Season", "Ramadan", "Eid al-Fitr", "School Summer", "Regular" |
| `App_Version_Period` | Quarterly period | "2023Q1", "2023Q2" |
| `gender_final`* | Predicted gender | "Male", "Female", "unknown" |

*Only if gender prediction is enabled

---

## Usage Examples

### Basic Usage (without gender)
```python
from alharam_analytics.pipeline import PreprocessingPipeline

pipeline = PreprocessingPipeline()
df = pipeline.run("data/raw/dataset.xlsx")
pipeline.save(df, "data/processed/dataset_clean.xlsx")
```

### With Gender Prediction
```python
pipeline = PreprocessingPipeline(include_gender_prediction=True)
df = pipeline.run("data/raw/dataset.xlsx")
```

### Run Specific Steps Only
```python
pipeline = PreprocessingPipeline()
df = pipeline.run(
    "data/raw/dataset.xlsx",
    steps=["username", "language", "period"]  # Only these steps
)
```

### CLI Usage
```bash
# Basic
python scripts/run_preprocessing.py -i data/raw/dataset.xlsx -o data/processed/output.xlsx

# With gender prediction
python scripts/run_preprocessing.py -i data/raw/dataset.xlsx -o data/processed/output.xlsx --gender

# Quiet mode
python scripts/run_preprocessing.py -i data/raw/dataset.xlsx -o data/processed/output.xlsx -q
```

---

## Configuration

Edit `config/pipeline_config.yaml` to customize:
- Column names
- Confidence thresholds
- Output column names
- Enable/disable specific features
