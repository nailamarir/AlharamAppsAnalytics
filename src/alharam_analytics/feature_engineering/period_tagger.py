"""
Period tagging for reviews based on Saudi calendar events.

Tags reviews with special periods like Hajj season, Ramadan, school holidays, etc.
"""

from datetime import date
from typing import Optional

import pandas as pd

try:
    from hijri_converter import convert
    HIJRI_AVAILABLE = True
except ImportError:
    HIJRI_AVAILABLE = False


# KSA School Summer Windows (inclusive dates)
KSA_SCHOOL_SUMMER = [
    (date(2012, 6, 6), date(2012, 9, 1)),
    (date(2013, 6, 5), date(2013, 8, 31)),
    (date(2014, 6, 4), date(2014, 8, 30)),
    (date(2015, 6, 4), date(2015, 8, 22)),
    (date(2016, 5, 26), date(2016, 9, 19)),
    (date(2017, 5, 25), date(2017, 9, 17)),
    (date(2018, 5, 24), date(2018, 9, 2)),
    (date(2019, 5, 30), date(2019, 9, 1)),
    (date(2020, 3, 9), date(2020, 8, 29)),   # COVID affected
    (date(2021, 5, 20), date(2021, 8, 29)),
    (date(2022, 6, 30), date(2022, 8, 28)),
    (date(2023, 6, 22), date(2023, 8, 20)),
    (date(2024, 6, 10), date(2024, 8, 14)),
    (date(2025, 6, 26), date(2025, 8, 24)),
]


def _is_in_school_summer(d: date) -> bool:
    """Check if date falls within KSA school summer vacation."""
    for start, end in KSA_SCHOOL_SUMMER:
        if start <= d <= end:
            return True
    return False


def _is_hajj_season(d: date) -> bool:
    """
    Check if date falls within Hajj season.

    Hajj occurs from 8th to 13th Dhul Hijjah (month 12 in Hijri calendar).
    We extend the window slightly for travel preparation.
    """
    if not HIJRI_AVAILABLE:
        return False

    try:
        hijri = convert.Gregorian(d.year, d.month, d.day).to_hijri()
        # Dhul Hijjah is month 12, Hajj days are roughly 8-13
        # We use a wider window (1-15) to capture preparation and travel
        return hijri.month == 12 and 1 <= hijri.day <= 15
    except Exception:
        return False


def _is_ramadan(d: date) -> bool:
    """Check if date falls within Ramadan (month 9 in Hijri calendar)."""
    if not HIJRI_AVAILABLE:
        return False

    try:
        hijri = convert.Gregorian(d.year, d.month, d.day).to_hijri()
        return hijri.month == 9
    except Exception:
        return False


def _is_eid_al_fitr(d: date) -> bool:
    """Check if date falls within Eid al-Fitr (1-3 Shawwal)."""
    if not HIJRI_AVAILABLE:
        return False

    try:
        hijri = convert.Gregorian(d.year, d.month, d.day).to_hijri()
        return hijri.month == 10 and 1 <= hijri.day <= 3
    except Exception:
        return False


def _is_eid_al_adha(d: date) -> bool:
    """Check if date falls within Eid al-Adha (10-13 Dhul Hijjah)."""
    if not HIJRI_AVAILABLE:
        return False

    try:
        hijri = convert.Gregorian(d.year, d.month, d.day).to_hijri()
        return hijri.month == 12 and 10 <= hijri.day <= 13
    except Exception:
        return False


def tag_period(d: date) -> str:
    """
    Tag a date with its corresponding special period.

    Priority order:
    1. Hajj Season
    2. Eid al-Adha
    3. Eid al-Fitr
    4. Ramadan
    5. School Summer
    6. Regular

    Args:
        d: Date to tag

    Returns:
        Period name string
    """
    if _is_hajj_season(d):
        return "Hajj Season"
    if _is_eid_al_adha(d):
        return "Eid al-Adha"
    if _is_eid_al_fitr(d):
        return "Eid al-Fitr"
    if _is_ramadan(d):
        return "Ramadan"
    if _is_in_school_summer(d):
        return "School Summer"
    return "Regular"


class PeriodTagger:
    """
    Tags DataFrame rows with special period information.

    Attributes:
        date_column: Name of the datetime column
        output_column: Name for the period column
    """

    def __init__(
        self,
        date_column: str = "Review Date",
        output_column: str = "period"
    ):
        self.date_column = date_column
        self.output_column = output_column

    def fit(self, df: pd.DataFrame) -> "PeriodTagger":
        """Fit method (no-op, for sklearn compatibility)."""
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform DataFrame by adding period tags.

        Args:
            df: Input DataFrame with date column

        Returns:
            DataFrame with added period column
        """
        df = df.copy()

        def safe_tag(dt):
            if pd.isna(dt):
                return "Unknown"
            try:
                d = dt.date() if hasattr(dt, "date") else dt
                return tag_period(d)
            except Exception:
                return "Unknown"

        df[self.output_column] = df[self.date_column].apply(safe_tag)

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)

    def add_quarter_period(
        self,
        df: pd.DataFrame,
        output_column: str = "App_Version_Period"
    ) -> pd.DataFrame:
        """
        Add quarterly period column for app version tracking.

        Args:
            df: Input DataFrame
            output_column: Name for the quarter column

        Returns:
            DataFrame with added quarter column
        """
        df = df.copy()
        df[output_column] = pd.to_datetime(df[self.date_column]).dt.to_period("Q").astype(str)
        return df
