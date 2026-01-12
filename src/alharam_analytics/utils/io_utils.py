"""
I/O utilities for loading and saving data.
"""

from pathlib import Path
from typing import Union

import pandas as pd


def load_data(
    file_path: Union[str, Path],
    **kwargs
) -> pd.DataFrame:
    """
    Load data from various file formats.

    Supports: .xlsx, .xls, .csv, .parquet, .json

    Args:
        file_path: Path to the data file
        **kwargs: Additional arguments passed to pandas reader

    Returns:
        Loaded DataFrame
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    readers = {
        ".xlsx": pd.read_excel,
        ".xls": pd.read_excel,
        ".csv": pd.read_csv,
        ".parquet": pd.read_parquet,
        ".json": pd.read_json,
    }

    if suffix not in readers:
        raise ValueError(f"Unsupported file format: {suffix}")

    return readers[suffix](file_path, **kwargs)


def save_data(
    df: pd.DataFrame,
    file_path: Union[str, Path],
    index: bool = False,
    **kwargs
) -> None:
    """
    Save DataFrame to various file formats.

    Supports: .xlsx, .xls, .csv, .parquet, .json

    Args:
        df: DataFrame to save
        file_path: Output file path
        index: Whether to include index (default: False)
        **kwargs: Additional arguments passed to pandas writer
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    # Ensure parent directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if suffix in [".xlsx", ".xls"]:
        df.to_excel(file_path, index=index, **kwargs)
    elif suffix == ".csv":
        df.to_csv(file_path, index=index, **kwargs)
    elif suffix == ".parquet":
        df.to_parquet(file_path, index=index, **kwargs)
    elif suffix == ".json":
        df.to_json(file_path, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")
