"""
Application name normalization and standardization.
"""

import pandas as pd


# Default mapping for app name variations
DEFAULT_APP_NAME_MAP = {
    "نسك": "Nusuk نسك",
    "حافلات مكه": "حافلات مكة",
    "توكلنا": "tawakkalna",
    "صحتي": "صحتي _ Sehhaty",
    "hhr-train": "قطار الحرمين",
    "المطوف - مناسك الحج والعمرة": "المطوف-مناسك الحج والعمرة",
    "الحج ثلاثي ابعاد": "الحج ثلاثي الابعاد",
    "إرشاد لإدارة خدمات شركات الحج": "ارشاد",
}


def normalize_app_name(
    name: str,
    name_map: dict = None
) -> str:
    """
    Normalize application name to standard form.

    Args:
        name: Raw application name
        name_map: Optional custom mapping dictionary

    Returns:
        Normalized application name
    """
    if name_map is None:
        name_map = DEFAULT_APP_NAME_MAP

    return name_map.get(name, name)


class AppNameNormalizer:
    """
    Normalizes application names in DataFrames.

    Attributes:
        column_name: Name of the application name column
        name_map: Mapping of variations to standard names
    """

    def __init__(
        self,
        column_name: str = "Application Name",
        name_map: dict = None
    ):
        self.column_name = column_name
        self.name_map = name_map or DEFAULT_APP_NAME_MAP

    def fit(self, df: pd.DataFrame) -> "AppNameNormalizer":
        """Fit method (no-op, for sklearn compatibility)."""
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform DataFrame by normalizing app names.

        Args:
            df: Input DataFrame with app name column

        Returns:
            DataFrame with normalized app names
        """
        df = df.copy()
        df[self.column_name] = df[self.column_name].apply(
            lambda x: normalize_app_name(x, self.name_map)
        )
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)

    def add_custom_mapping(self, original: str, normalized: str):
        """Add a custom name mapping."""
        self.name_map[original] = normalized

    def add_supported_platforms_count(
        self,
        df: pd.DataFrame,
        device_column: str = "Device Type",
        output_column: str = "SupportedPlatforms"
    ) -> pd.DataFrame:
        """
        Add count of supported platforms per app.

        Args:
            df: Input DataFrame
            device_column: Column containing device type
            output_column: Name for the platform count column

        Returns:
            DataFrame with platform count merged
        """
        df = df.copy()

        platform_count = (
            df.groupby(self.column_name)[device_column]
            .nunique()
            .reset_index()
            .rename(columns={device_column: output_column})
        )

        df = df.merge(platform_count, on=self.column_name, how="left")
        return df
