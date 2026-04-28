import pandas as pd
from .types import CleaningConfig
from typing import Tuple

def fill_short_gaps(df: pd.DataFrame, config: CleaningConfig) -> Tuple[pd.DataFrame, int]:
    """
    Fills small missing data gaps using interpolation.
    
    Args:
        df: Input DataFrame with DatetimeIndex.
        config: Cleaning configuration (max_gap_minutes).
        
    Returns:
        Tuple[pd.DataFrame, int]: Interpolated DataFrame and count of filled values.
    """
    df = df.copy()
    
    # Calculate how many rows correspond to max_gap_minutes
    # Heuristic: find the most frequent interval in the index
    if len(df.index) < 2:
        return df, 0
        
    freq = (df.index[1] - df.index[0]).total_seconds() / 60
    if freq <= 0:
        return df, 0
        
    limit = int(config.max_gap_minutes / freq)
    
    initial_nan = df.isna().sum().sum()
    
    # Linear interpolation with row limit
    # We use 'forward' to avoid backfilling startup data into night shutdown periods
    df = df.interpolate(method='linear', limit=limit, limit_direction='forward')
    
    final_nan = df.isna().sum().sum()
    interpolated_count = int(initial_nan - final_nan)
    
    return df, interpolated_count
