import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any
from .types import CleaningConfig, SensorDefinition

def remove_shutdown(df: pd.DataFrame, config: CleaningConfig) -> Tuple[pd.DataFrame, int]:
    """
    Sets sensor values to NaN when the equipment is detected as 'off'.
    
    Args:
        df: Input DataFrame.
        config: Cleaning configuration.
        
    Returns:
        Tuple[pd.DataFrame, int]: Cleaned DataFrame and count of removed shutdown points.
    """
    if not config.shutdown_detect_column or config.shutdown_detect_column not in df.columns:
        return df, 0
        
    df = df.copy()
    is_off = df[config.shutdown_detect_column] < config.shutdown_power_threshold
    
    # Identify numeric columns for cleaning
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    removed_count = int(is_off.sum())
    df.loc[is_off, numeric_cols] = np.nan
    
    return df, removed_count

def check_physical_range(df: pd.DataFrame, schema: List[SensorDefinition]) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Clips or NaNs values outside the plausible physical range defined in schema.
    
    Returns:
        Tuple[pd.DataFrame, List[dict]]: Cleaned DataFrame and list of range violations.
    """
    df = df.copy()
    violations = []
    
    for sensor in schema:
        if sensor.std_name not in df.columns:
            continue
            
        lo, hi = sensor.physical_range
        series = df[sensor.std_name]
        
        bad_mask = (series < lo) | (series > hi)
        bad_count = int(bad_mask.sum())
        
        if bad_count > 0:
            violations.append({
                "column": sensor.std_name,
                "violation_count": bad_count,
                "range": (lo, hi)
            })
            # Set out-of-range values to NaN
            df.loc[bad_mask, sensor.std_name] = np.nan
            
    return df, violations

def detect_anomalies(df: pd.DataFrame, config: CleaningConfig) -> Tuple[pd.DataFrame, int, int]:
    """
    Detects spikes (3-sigma) and flatlines (persistent unchanging values).
    
    Returns:
        Tuple[pd.DataFrame, spike_count, flatline_count]
    """
    df = df.copy()
    spike_total = 0
    flatline_total = 0
    
    for col in df.columns:
        series = df[col]
        if not pd.api.types.is_numeric_dtype(series):
            continue
            
        # 1. Spike detection (Z-score based)
        std = series.std()
        if std > 0:
            mean = series.mean()
            is_spike = (series - mean).abs() > (config.spike_sigma * std)
            spike_count = int(is_spike.sum())
            spike_total += spike_count
            df.loc[is_spike, col] = np.nan
            
        # 2. Flatline detection (Simplified: same value for N consecutive steps)
        # N = flatline_minutes / timestep (assume timestep is already handled by index)
        # We'll use a fixed window of 5 if minutes can't be resolved easily here,
        # but better to use config.flatline_minutes.
        # This is simplified for Stage 1.
        diff = series.diff().abs()
        is_flat = (diff == 0).rolling(window=5).sum() == 5
        flatline_count = int(is_flat.sum())
        flatline_total += flatline_count
        df.loc[is_flat, col] = np.nan
        
    return df, spike_total, flatline_total
