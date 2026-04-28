from __future__ import annotations

import numpy as np
import pandas as pd


class CompressedAirPredictionPart:
    def get_prediction_targets(self) -> list[str]:
        return ["total_power"]

    def build_features(self, df: pd.DataFrame, target: str) -> pd.DataFrame:
        out = df.copy()
        out.index = pd.to_datetime(out.index, errors="coerce")
        out = out[out.index.notnull()].sort_index()
        out["hour_sin"] = np.sin(2 * np.pi * out.index.hour / 24)
        out["hour_cos"] = np.cos(2 * np.pi * out.index.hour / 24)
        out["target_lag_1"] = out[target].shift(1)
        out["target_roll_mean_4"] = out[target].shift(1).rolling(4).mean()
        return out.dropna()

    def get_feature_columns(self, df: pd.DataFrame, target: str) -> list[str]:
        return [c for c in df.columns if c != target]

    def get_model_preset(self) -> dict:
        return {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05}
