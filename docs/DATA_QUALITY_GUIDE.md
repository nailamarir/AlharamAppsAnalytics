# Data Quality Guide for AlHaram Analytics

## Current Dataset Issues & Solutions

Based on your notebooks, here are the data quality challenges and my suggestions:

---

## 1. Missing Values Strategy

### Current State
```
Review ID              2960 missing
Review Title          54761 missing
Review Text              26 missing
Helpful Votes          2960 missing
Developer Response    37811 missing
Reply Date            38384 missing
```

### Recommendations

| Column | Strategy | Rationale |
|--------|----------|-----------|
| `Review ID` | Generate sequential IDs | Required for tracking |
| `Review Title` | Keep as NaN or fill with "" | Many reviews don't have titles |
| `Review Text` | **Drop rows** with missing text | Core data - can't analyze empty reviews |
| `User Name` | Fill with "Anonymous" | ✅ Already implemented |
| `Developer Response` | Keep as NaN | Absence is meaningful (no response) |
| `Reply Date` | Keep as NaN | Tied to Developer Response |

### Suggested Enhancement
```python
# Add to pipeline
def handle_missing_values(df):
    # Drop reviews without text (can't analyze)
    df = df.dropna(subset=['Review Text'])

    # Generate Review IDs if missing
    df['Review ID'] = df['Review ID'].fillna(
        pd.Series(range(1, len(df) + 1))
    )

    # Create flag for developer response
    df['has_developer_response'] = df['Developer Response'].notna()

    return df
```

---

## 2. Gender Prediction Quality

### Current Issues
- Two models sometimes disagree
- Many "unknown" results (especially for Arabic names)
- Confidence scores vary widely

### Recommendations

#### A. Improve Training Data
Create a **ground truth dataset** for validation:

```python
# Sample 500-1000 names for manual labeling
sample_names = df['clean_name'].drop_duplicates().sample(1000)
sample_names.to_csv('data/validation/names_to_label.csv')
```

Then manually label them to evaluate model accuracy.

#### B. Add Arabic Name Dictionary
Many Arabic names are culturally unambiguous:

```python
ARABIC_MALE_NAMES = {
    'محمد', 'أحمد', 'عبدالله', 'خالد', 'عمر', 'علي', 'سعود',
    'فهد', 'سلطان', 'ناصر', 'فيصل', 'عبدالرحمن', 'طلال'
}

ARABIC_FEMALE_NAMES = {
    'فاطمة', 'عائشة', 'مريم', 'نورة', 'سارة', 'هند', 'لمى',
    'ريم', 'دانة', 'غادة', 'منى', 'أمل', 'هيا'
}

def lookup_arabic_gender(name):
    first_word = name.split()[0] if name else ""
    if first_word in ARABIC_MALE_NAMES:
        return "Male"
    if first_word in ARABIC_FEMALE_NAMES:
        return "Female"
    return None  # Fall back to model
```

#### C. Confidence Tiers
```python
def categorize_gender_confidence(row):
    score = row['pred_gender_score']
    if score >= 0.95:
        return 'high_confidence'
    elif score >= 0.80:
        return 'medium_confidence'
    elif score >= 0.60:
        return 'low_confidence'
    else:
        return 'unreliable'
```

---

## 3. Text Quality Improvements

### A. Review Text Cleaning
```python
def clean_review_text(text):
    if pd.isna(text):
        return ""

    text = str(text)

    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove repeated characters (e.g., "sooooo" → "so")
    text = re.sub(r'(.)\1{3,}', r'\1\1', text)

    return text
```

### B. Spam/Bot Detection
```python
def detect_potential_spam(row):
    flags = []

    text = str(row['Review Text']).lower()

    # Very short reviews
    if len(text) < 10:
        flags.append('too_short')

    # Excessive punctuation
    if text.count('!') > 5:
        flags.append('excessive_punctuation')

    # Repeated patterns
    if re.search(r'(.{10,})\1', text):
        flags.append('repeated_content')

    # Generic spam phrases
    spam_phrases = ['click here', 'visit my', 'check out', 'download now']
    if any(phrase in text for phrase in spam_phrases):
        flags.append('spam_phrase')

    return flags if flags else None
```

---

## 4. Suggested New Features

### A. Sentiment Indicators (Simple)
```python
# Quick sentiment from rating
def rating_sentiment(rating):
    if rating >= 4:
        return 'positive'
    elif rating == 3:
        return 'neutral'
    else:
        return 'negative'
```

### B. Review Length Categories
```python
def review_length_category(text):
    length = len(str(text))
    if length < 50:
        return 'short'
    elif length < 200:
        return 'medium'
    else:
        return 'long'
```

### C. Time-Based Features
```python
def add_time_features(df):
    df['review_weekday'] = df['Review Date'].dt.day_name()
    df['is_weekend'] = df['Review Date'].dt.dayofweek >= 5
    df['review_hour_category'] = pd.cut(
        df['Review Hour'],
        bins=[0, 6, 12, 18, 24],
        labels=['night', 'morning', 'afternoon', 'evening']
    )
    return df
```

### D. User Engagement Score
```python
def calculate_engagement_score(row):
    score = 0

    # Longer reviews = more engaged
    text_length = len(str(row['Review Text']))
    score += min(text_length / 100, 3)  # Max 3 points

    # Has title = more effort
    if pd.notna(row['Review Title']):
        score += 1

    # Not anonymous = more invested
    if row['clean_name'] != 'Anonymous':
        score += 1

    return round(score, 2)
```

---

## 5. Data Validation Checks

Add these checks to your pipeline:

```python
def validate_dataset(df):
    issues = []

    # Check for required columns
    required = ['Review Date', 'Review Text', 'Rating', 'Application Name']
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")

    # Check rating range
    if df['Rating'].min() < 1 or df['Rating'].max() > 5:
        issues.append("Rating values outside 1-5 range")

    # Check date range
    if df['Review Date'].min().year < 2010:
        issues.append("Suspicious dates before 2010")

    # Check for duplicates
    dup_count = df.duplicated(subset=['Review Text', 'User Name', 'Review Date']).sum()
    if dup_count > 0:
        issues.append(f"Found {dup_count} potential duplicate reviews")

    # Report
    if issues:
        print("⚠️  Data Quality Issues:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("✅ All validation checks passed")

    return issues
```

---

## 6. Recommended Final Schema

After all preprocessing, your quality dataset should have:

```
CORE FIELDS (Original)
├── Review ID (generated if missing)
├── Review Date
├── Review Text
├── Rating
├── Application Name
├── Platform
└── User Name

CLEANED FIELDS
├── clean_name
├── clean_review_text
└── has_developer_response (boolean)

ENRICHMENT FIELDS
├── language (Arabic/English/Mixed/Unknown)
├── Device Type (iOS/Android)
├── Service_Type (Health/Reservation/Government/Religious/Others)
├── period (Hajj/Ramadan/Eid/School Summer/Regular)
├── App_Version_Period (2023Q1, etc.)
├── gender_final (Male/Female/unknown)
└── gender_confidence (high/medium/low/unreliable)

ANALYTICS FIELDS
├── review_length_category (short/medium/long)
├── rating_sentiment (positive/neutral/negative)
├── review_weekday
├── is_weekend
├── engagement_score
└── spam_flags (list or null)
```

---

## 7. Quality Metrics to Track

Monitor these metrics over time:

| Metric | Target | Current |
|--------|--------|---------|
| Missing Review Text | < 1% | ~0.04% ✅ |
| Unknown Language | < 5% | TBD |
| Unknown Gender | < 20% | ~10% ✅ |
| Anonymous Users | < 10% | TBD |
| Potential Duplicates | 0% | TBD |
| Valid Date Range | 100% | TBD |

---

## Next Steps

1. **Implement validation checks** in the pipeline
2. **Create ground truth sample** for gender accuracy testing
3. **Add Arabic name dictionary** for better gender prediction
4. **Build data quality dashboard** to monitor metrics
5. **Document any manual corrections** for reproducibility

Would you like me to implement any of these enhancements?
