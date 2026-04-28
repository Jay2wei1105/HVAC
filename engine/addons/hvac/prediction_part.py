import numpy as np
import pandas as pd
from typing import List

try:
    import holidays
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    holidays = None


def calc_wet_bulb(temp_c: pd.Series, rh_pct: pd.Series) -> pd.Series:
    """Approximate wet-bulb temperature in Celsius using the Stull formula."""
    t = temp_c.astype(float)
    rh = rh_pct.astype(float)
    return (
        t * np.arctan(0.151977 * np.sqrt(rh + 8.313659))
        + np.arctan(t + rh)
        - np.arctan(rh - 1.676331)
        + 0.00391838 * np.power(rh, 1.5) * np.arctan(0.023101 * rh)
        - 4.686035
    )


class HVACPredictionPart:
    """Stage 2 prediction features for HVAC regression targets."""

    def get_prediction_targets(self) -> List[str]:
        return ["total_power", "chw_delta_t", "system_kw_per_rt"]

    def build_features(self, df: pd.DataFrame, target: str) -> pd.DataFrame:
        df = df.copy()
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[df.index.notnull()].sort_index()

        df["hour"] = df.index.hour
        df["dayofweek"] = df.index.dayofweek
        df["month"] = df.index.month
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

        if holidays is not None:
            tw_holidays = holidays.TW()
            df["is_holiday"] = df.index.map(lambda x: 1 if x in tw_holidays else 0)
        else:
            df["is_holiday"] = 0
        df["is_weekend"] = df.index.dayofweek.map(lambda x: 1 if x >= 5 else 0)

        if "temp" in df.columns and "rh" in df.columns:
            df["enthalpy"] = 1.006 * df["temp"] + (df["rh"] / 100) * 0.001 * (
                2501 + 1.86 * df["temp"]
            )
            df["wb_temp"] = calc_wet_bulb(df["temp"], df["rh"])

        for lag in [1, 2, 4, 12]:
            df[f"target_lag_{lag}"] = df[target].shift(lag)

        df["target_roll_mean_4"] = df[target].shift(1).rolling(window=4).mean()
        df["target_roll_std_4"] = df[target].shift(1).rolling(window=4).std()

        return df.dropna()

    def get_feature_columns(self, df: pd.DataFrame, target: str) -> List[str]:
        exclude = {target, "hour", "dayofweek", "month"}
        return [c for c in df.columns if c not in exclude]

    def get_model_preset(self) -> dict:
        return {
            "n_estimators": 800,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
        }
