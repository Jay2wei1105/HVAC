# engine/core/mapper.py
"""
Column Mapper — PDF §2
Three-layer confidence: high (keyword hit), medium (value hint), low (missing).
"""
import pandas as pd
from dataclasses import dataclass, field
from typing import Literal
from .types import SensorDefinition


ConfidenceLevel = Literal["high", "medium", "low"]


@dataclass
class MappingResult:
    """Column mapping with per-sensor confidence levels."""
    mapping: dict[str, str]                     # std_name -> original col
    confidence: dict[str, ConfidenceLevel]      # std_name -> confidence
    unmatched: list[str]                        # std_names that had no match


def auto_map(
    df: pd.DataFrame,
    schema: list[SensorDefinition],
) -> MappingResult:
    """Layer 1 keyword → Layer 2 value_hint. Returns mapping with confidence."""
    mapping: dict[str, str] = {}
    confidence: dict[str, ConfidenceLevel] = {}
    used_cols: set[str] = set()

    # Layer 1 — keyword matching (HIGH confidence)
    for sensor in schema:
        for col in df.columns:
            if col in used_cols:
                continue
            if any(kw.lower() in col.lower() for kw in sensor.keywords):
                mapping[sensor.std_name] = col
                confidence[sensor.std_name] = "high"
                used_cols.add(col)
                break

    # Layer 2 — value hint matching (MEDIUM confidence)
    for sensor in schema:
        if sensor.std_name in mapping or sensor.value_hint is None:
            continue
        lo, hi = sensor.value_hint
        for col in df.columns:
            if col in used_cols:
                continue
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if series.empty:
                continue
            if lo <= series.min() and series.max() <= hi:
                mapping[sensor.std_name] = col
                confidence[sensor.std_name] = "medium"
                used_cols.add(col)
                break

    # Unmatched sensors
    unmatched = [s.std_name for s in schema if s.std_name not in mapping]
    for name in unmatched:
        confidence[name] = "low"

    return MappingResult(mapping=mapping, confidence=confidence, unmatched=unmatched)


def apply_mapping(df: pd.DataFrame, col_map: dict[str, str]) -> pd.DataFrame:
    """Renames columns based on the mapping and drops unmapped columns."""
    df = df.copy()

    # Detect time column to preserve it
    from .time_index import _detect_time_column
    time_col = _detect_time_column(df)

    # Invert mapping for rename: original -> std
    inv_map = {v: k for k, v in col_map.items()}
    df = df.rename(columns=inv_map)

    # Keep mapped columns + original time column
    cols_to_keep = list(col_map.keys())
    if time_col and time_col in df.columns:
        if time_col not in inv_map.values():
            cols_to_keep.append(time_col)

    return df[cols_to_keep]


def assert_required_columns(
    df: pd.DataFrame, schema: list[SensorDefinition],
) -> None:
    """Raise if required columns are missing after mapping."""
    missing = [s.std_name for s in schema
               if s.required and s.std_name not in df.columns]
    if missing:
        raise ValueError(f"Required columns missing after mapping: {missing}")
