import pandas as pd
from typing import Union
from pathlib import Path

def load_csv(path: Union[str, Path]) -> pd.DataFrame:
    """
    Loads raw CSV data. Simple wrapper for pandas.read_csv to allow future Polars integration.
    
    Args:
        path: Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded data.
    """
    # Using low_memory=False to avoid DtypeWarning on large files
    return pd.read_csv(path, low_memory=False)
