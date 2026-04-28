from __future__ import annotations

import numpy as np
import pandas as pd

from .prediction_part import calc_wet_bulb


class HVACQDemandPart:
    """Q-demand safety-bound prediction features for Stage 3 advisory."""

    def get_prediction_targets(self) -> list[str]:
        return ["q_delivered_kw"]

    def build_features(
        self,
        df: pd.DataFrame,
        target: str = "q_delivered_kw",
    ) -> pd.DataFrame:
        feat_df = df.copy()
        feat_df.index = pd.to_datetime(feat_df.index, errors="coerce")
        feat_df = feat_df[feat_df.index.notnull()].sort_index()

        hour = feat_df.index.hour
        dow = feat_df.index.dayofweek
        feat_df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        feat_df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        feat_df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        feat_df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
        feat_df["is_business_hour"] = ((hour >= 8) & (hour < 18) & (dow < 5)).astype(int)

        outdoor_temp_col, outdoor_rh_col = self._resolve_weather_columns(feat_df)
        if outdoor_temp_col and outdoor_rh_col:
            feat_df["wet_bulb"] = calc_wet_bulb(
                feat_df[outdoor_temp_col],
                feat_df[outdoor_rh_col],
            )
        if outdoor_temp_col:
            feat_df["outdoor_temp"] = feat_df[outdoor_temp_col].astype(float)
            roll_window = min(16, max(4, len(feat_df) // 8)) if len(feat_df) else 16
            feat_df["outdoor_temp_roll_4h"] = feat_df["outdoor_temp"].rolling(
                roll_window,
                min_periods=max(2, min(roll_window, 4)),
            ).mean()
        if outdoor_rh_col and "outdoor_rh" not in feat_df.columns:
            feat_df["outdoor_rh"] = feat_df[outdoor_rh_col].astype(float)

        candidate_lags = (4, 16, 96)
        usable_lags = [lag for lag in candidate_lags if len(feat_df) > lag + 4]
        if not usable_lags and len(feat_df) > 8:
            usable_lags = [4]
        for lag in usable_lags:
            feat_df[f"q_lag_{lag}"] = feat_df[target].shift(lag)

        # Keep rows with a valid target. LightGBM can consume NaNs in optional
        # features, so we should not throw away the whole frame for sparse columns.
        feat_df = feat_df[feat_df[target].notna()].copy()

        if "is_steady" in feat_df.columns:
            feat_df["is_steady"] = feat_df["is_steady"].fillna(False).astype(bool)

        return feat_df

    def get_feature_columns(
        self,
        df: pd.DataFrame,
        target: str = "q_delivered_kw",
    ) -> list[str]:
        excluded = {target, "is_steady"}
        return [
            c
            for c in df.columns
            if c not in excluded and not c.startswith("_") and df[c].notna().any()
        ]

    def get_model_preset(self) -> dict[str, float | int | str]:
        return {
            "objective": "quantile",
            "alpha": 0.9,
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "random_state": 42,
        }

    @staticmethod
    def _resolve_weather_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
        temp_candidates = ("outdoor_temp", "temp")
        rh_candidates = ("outdoor_rh", "rh")

        temp_col = next((col for col in temp_candidates if col in df.columns), None)
        rh_col = next((col for col in rh_candidates if col in df.columns), None)
        return temp_col, rh_col
