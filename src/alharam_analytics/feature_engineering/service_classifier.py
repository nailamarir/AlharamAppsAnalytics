"""
Service type classification for AlHaram applications.
"""

import pandas as pd


# Default service type mapping
DEFAULT_SERVICE_MAPPING = {
    # Health-related apps
    "صحتي": "Health",
    "صحتي _ Sehhaty": "Health",
    "أسعفني": "Health",

    # Reservation / Travel / Transport apps
    "حافلات مكة": "Reservation",
    "حافلات مكه": "Reservation",
    "قطار الحرمين": "Reservation",
    "عامرة": "Reservation",
    "زيارة الحرم": "Reservation",
    "زائرون": "Reservation",
    "تنقل": "Reservation",
    "تروية": "Reservation",
    "زوار مكة": "Reservation",

    # Government Services
    "توكلنا": "Government Services",
    "tawakkalna": "Government Services",
    "إرشاد لإدارة خدمات شركات الحج": "Government Services",
    "ارشاد": "Government Services",
    "Hajj and Umrah Navigator": "Government Services",
    "نسك": "Government Services",
    "Nusuk نسك": "Government Services",

    # Religious apps
    "مكتشف القبله": "Religious",
    "فاذكروني": "Religious",
    "مصحف الحرمين": "Religious",

    # Hajj / Umrah guides and other apps
    "دليل الحج والعمره ": "Others",
    "منارة الحرمين ": "Others",
    "المطوف ": "Others",
    "المطوف-مناسك الحج والعمرة": "Others",
    "AlMaqsad": "Others",
    "رفيق الحاج": "Others",
    "الحج ثلاثي الابعاد": "Others",
    "مخطط مناسك الحج والعمرة ": "Others",
}


def classify_service_type(
    app_name: str,
    service_map: dict = None
) -> str:
    """
    Classify application into service type category.

    Args:
        app_name: Application name
        service_map: Optional custom mapping dictionary

    Returns:
        Service type category
    """
    if service_map is None:
        service_map = DEFAULT_SERVICE_MAPPING

    return service_map.get(app_name, "Others")


class ServiceClassifier:
    """
    Classifies applications by service type.

    Attributes:
        app_column: Name of the application name column
        output_column: Name for the service type column
        service_map: Mapping of app names to service types
    """

    def __init__(
        self,
        app_column: str = "Application Name",
        output_column: str = "Service_Type",
        service_map: dict = None
    ):
        self.app_column = app_column
        self.output_column = output_column
        self.service_map = service_map or DEFAULT_SERVICE_MAPPING

    def fit(self, df: pd.DataFrame) -> "ServiceClassifier":
        """Fit method (no-op, for sklearn compatibility)."""
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform DataFrame by classifying service types.

        Args:
            df: Input DataFrame with app name column

        Returns:
            DataFrame with added service type column
        """
        df = df.copy()
        df[self.output_column] = df[self.app_column].apply(
            lambda x: classify_service_type(x, self.service_map)
        )
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)

    def add_custom_mapping(self, app_name: str, service_type: str):
        """Add a custom app-to-service mapping."""
        self.service_map[app_name] = service_type

    def get_service_distribution(self, df: pd.DataFrame) -> pd.Series:
        """Get distribution of service types in DataFrame."""
        if self.output_column not in df.columns:
            df = self.transform(df)
        return df[self.output_column].value_counts()
