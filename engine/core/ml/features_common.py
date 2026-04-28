# engine/core/ml/features_common.py
"""
Universal Feature Registry — PDF §4
6 families × ~30 features, each with required_columns.
Build what you can, skip what you can't, report coverage.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class FeatureSpec:
    """Specification for a single computed feature."""
    family: str
    required: list[str]
    builder: Callable[[pd.DataFrame, str], pd.Series]
    display_name: str = ""


@dataclass
class CoverageReport:
    """Result of feature building with coverage stats."""
    coverage: float
    built: list[str]
    skipped: dict[str, list[str]]  # feature_name -> missing columns


# ── Builder functions ──────────────────────────────────────

def _hour_sin(df: pd.DataFrame, _t: str) -> pd.Series:
    return np.sin(2 * np.pi * df.index.hour / 24)

def _hour_cos(df: pd.DataFrame, _t: str) -> pd.Series:
    return np.cos(2 * np.pi * df.index.hour / 24)

def _dow_sin(df: pd.DataFrame, _t: str) -> pd.Series:
    return np.sin(2 * np.pi * df.index.dayofweek / 7)

def _dow_cos(df: pd.DataFrame, _t: str) -> pd.Series:
    return np.cos(2 * np.pi * df.index.dayofweek / 7)

def _month_sin(df: pd.DataFrame, _t: str) -> pd.Series:
    return np.sin(2 * np.pi * df.index.month / 12)

def _month_cos(df: pd.DataFrame, _t: str) -> pd.Series:
    return np.cos(2 * np.pi * df.index.month / 12)

def _is_weekend(df: pd.DataFrame, _t: str) -> pd.Series:
    return (df.index.dayofweek >= 5).astype(int)

def _chw_delta_t(df: pd.DataFrame, _t: str) -> pd.Series:
    return df["chw_return_temp"] - df["chw_supply_temp"]

def _wet_bulb(df: pd.DataFrame, _t: str) -> pd.Series:
    """Stull (2011) approximation."""
    t, rh = df["outdoor_temp"], df["outdoor_rh"]
    return (t * np.arctan(0.151977 * (rh + 8.313659) ** 0.5)
            + np.arctan(t + rh) - np.arctan(rh - 1.676331)
            + 0.00391838 * rh ** 1.5 * np.arctan(0.023101 * rh) - 4.686035)

def _enthalpy(df: pd.DataFrame, _t: str) -> pd.Series:
    t, rh = df["outdoor_temp"], df["outdoor_rh"]
    return 1.006 * t + (rh / 100) * 0.001 * (2501 + 1.86 * t)

def _cw_delta_t(df: pd.DataFrame, _t: str) -> pd.Series:
    return df["cw_return_temp"] - df["cw_supply_temp"]

def _approach_temp(df: pd.DataFrame, _t: str) -> pd.Series:
    return df["cw_supply_temp"] - df["outdoor_temp"]

def _make_lag(steps: int) -> Callable:
    def _fn(df: pd.DataFrame, t: str) -> pd.Series:
        return df[t].shift(steps)
    return _fn

def _make_roll_mean(window: int) -> Callable:
    def _fn(df: pd.DataFrame, t: str) -> pd.Series:
        return df[t].shift(1).rolling(window).mean()
    return _fn

def _make_roll_std(window: int) -> Callable:
    def _fn(df: pd.DataFrame, t: str) -> pd.Series:
        return df[t].shift(1).rolling(window).std()
    return _fn

def _oa_roll_mean_12(df: pd.DataFrame, _t: str) -> pd.Series:
    return df["outdoor_temp"].rolling(12).mean()

def _temp_x_rh(df: pd.DataFrame, _t: str) -> pd.Series:
    return df["outdoor_temp"] * df["outdoor_rh"]


# ── Registry ────────────────────────────────────────────────

FEATURE_REGISTRY: dict[str, FeatureSpec] = {
    # Family 1: Time Cycle
    "hour_sin":   FeatureSpec("time_cycle", [], _hour_sin, "小時 sin"),
    "hour_cos":   FeatureSpec("time_cycle", [], _hour_cos, "小時 cos"),
    "dow_sin":    FeatureSpec("time_cycle", [], _dow_sin, "星期 sin"),
    "dow_cos":    FeatureSpec("time_cycle", [], _dow_cos, "星期 cos"),
    "month_sin":  FeatureSpec("time_cycle", [], _month_sin, "月份 sin"),
    "month_cos":  FeatureSpec("time_cycle", [], _month_cos, "月份 cos"),
    "is_weekend": FeatureSpec("time_cycle", [], _is_weekend, "是否週末"),
    # Family 2: Weather Derived
    "chw_delta_t":  FeatureSpec("weather_derived",
                                ["chw_return_temp", "chw_supply_temp"],
                                _chw_delta_t, "冰水溫差"),
    "wet_bulb":     FeatureSpec("weather_derived",
                                ["outdoor_temp", "outdoor_rh"],
                                _wet_bulb, "濕球溫度"),
    "enthalpy":     FeatureSpec("weather_derived",
                                ["outdoor_temp", "outdoor_rh"],
                                _enthalpy, "空氣焓值"),
    "cw_delta_t":   FeatureSpec("weather_derived",
                                ["cw_return_temp", "cw_supply_temp"],
                                _cw_delta_t, "冷卻水溫差"),
    "approach_temp": FeatureSpec("weather_derived",
                                 ["cw_supply_temp", "outdoor_temp"],
                                 _approach_temp, "趨近溫度"),
    # Family 3: Target Lag
    "target_lag_1":  FeatureSpec("target_lag", [], _make_lag(1), "目標 t-15m"),
    "target_lag_2":  FeatureSpec("target_lag", [], _make_lag(2), "目標 t-30m"),
    "target_lag_4":  FeatureSpec("target_lag", [], _make_lag(4), "目標 t-1h"),
    "target_lag_12": FeatureSpec("target_lag", [], _make_lag(12), "目標 t-3h"),
    # Family 4: Target Rolling
    "target_roll_mean_4": FeatureSpec("target_rolling", [],
                                      _make_roll_mean(4), "目標 1h 均值"),
    "target_roll_std_4":  FeatureSpec("target_rolling", [],
                                      _make_roll_std(4), "目標 1h 標準差"),
    "target_roll_mean_12": FeatureSpec("target_rolling", [],
                                       _make_roll_mean(12), "目標 3h 均值"),
    # Family 5: Weather Rolling
    "oa_roll_mean_12": FeatureSpec("weather_rolling",
                                   ["outdoor_temp"],
                                   _oa_roll_mean_12, "外氣 3h 均溫"),
    # Family 6: Interaction
    "temp_x_rh": FeatureSpec("interaction",
                              ["outdoor_temp", "outdoor_rh"],
                              _temp_x_rh, "溫度×濕度"),
}


def build_features_from_registry(
    df: pd.DataFrame,
    target: str,
    registry: Optional[dict[str, FeatureSpec]] = None,
) -> tuple[pd.DataFrame, CoverageReport]:
    """Build all features whose required columns exist in df."""
    if registry is None:
        registry = FEATURE_REGISTRY

    # Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[df.index.notnull()].sort_index()

    result = df.copy()
    built: list[str] = []
    skipped: dict[str, list[str]] = {}

    for name, spec in registry.items():
        # target_lag / target_rolling need the target col to exist
        effective_req = [c for c in spec.required]
        if spec.family in ("target_lag", "target_rolling"):
            effective_req.append(target)

        missing = [c for c in effective_req if c not in result.columns]
        if missing:
            skipped[name] = missing
            continue
        try:
            result[name] = spec.builder(result, target)
            built.append(name)
        except Exception:
            skipped[name] = ["__build_error__"]

    coverage = len(built) / len(registry) if registry else 0
    report = CoverageReport(coverage=coverage, built=built, skipped=skipped)
    return result.dropna(), report
