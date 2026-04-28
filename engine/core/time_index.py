import pandas as pd
from typing import Optional

def build(df: pd.DataFrame, timestep_minutes: int) -> pd.DataFrame:
    """
    Parses the time column and rebuilds an equidistant time index.
    
    Args:
        df: Input DataFrame.
        timestep_minutes: Target time interval in minutes.
        
    Returns:
        pd.DataFrame: DataFrame with DatetimeIndex and resampled/reindexed rows.
    """
    df = df.copy()
    
    # 1. Detect time column
    time_col = _detect_time_column(df)
    if not time_col:
        raise ValueError("Could not detect a valid time column in the CSV.")
        
    # 2. Convert to datetime
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    df = df.dropna(subset=[time_col])
    
    # 3. Set index
    df = df.set_index(time_col).sort_index()
    
    # 4. Remove duplicates (take last if multiple entries for same timestamp)
    df = df[~df.index.duplicated(keep='last')]
    
    # 5. Reindex to equidistant grid
    start, end = df.index.min(), df.index.max()
    grid = pd.date_range(start=start, end=end, freq=f"{timestep_minutes}min")
    df = df.reindex(grid)
    
    return df

def _detect_time_column(df: pd.DataFrame) -> Optional[str]:
    """Simple heuristic to find the most likely time column."""
    candidates = ["time", "timestamp", "datetime", "date", "clock", "時間", "日期"]
    for col in df.columns:
        if any(cand in col.lower() for cand in candidates):
            return col
            
    # Fallback: check first few rows for datetime-like strings
    for col in df.columns:
        sample = df[col].head(5).astype(str)
        try:
            pd.to_datetime(sample, errors='raise')
            return col
        except:
            continue
            
    return None
