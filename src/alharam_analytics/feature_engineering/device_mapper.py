"""
Device type mapping from platform information.
"""

import pandas as pd


def map_device_type(platform: str) -> str:
    """
    Map platform string to device type.

    Args:
        platform: Platform string (e.g., "App Store", "Google Play")

    Returns:
        Device type: "iOS", "Android", or "Other"
    """
    if not isinstance(platform, str):
        return "Other"

    platform_lower = platform.lower()

    if "app store" in platform_lower or "ios" in platform_lower:
        return "iOS"
    elif "google play" in platform_lower or "android" in platform_lower:
        return "Android"
    else:
        return "Other"


class DeviceTypeMapper:
    """
    Maps platform column to device type.

    Attributes:
        platform_column: Name of the platform column
        output_column: Name for the device type column
    """

    def __init__(
        self,
        platform_column: str = "Platform",
        output_column: str = "Device Type"
    ):
        self.platform_column = platform_column
        self.output_column = output_column

    def fit(self, df: pd.DataFrame) -> "DeviceTypeMapper":
        """Fit method (no-op, for sklearn compatibility)."""
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform DataFrame by mapping device types.

        Args:
            df: Input DataFrame with platform column

        Returns:
            DataFrame with added device type column
        """
        df = df.copy()
        df[self.output_column] = df[self.platform_column].apply(map_device_type)
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)
