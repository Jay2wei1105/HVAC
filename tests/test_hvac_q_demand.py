from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.addons.hvac import HVACAddon
from engine.core.ml.q_demand_trainer import pinball_loss, train_q_demand_model
from engine.core.ml import q_demand_trainer as qd_trainer
from engine.core.ml import trainer as power_trainer


def _make_training_frame(n: int = 220) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=n, freq="15min")
    hour = index.hour.to_numpy()
    outdoor_temp = 28 + 4 * np.sin(2 * np.pi * hour / 24) + np.linspace(0, 1.5, n)
    outdoor_rh = 65 + 8 * np.cos(2 * np.pi * hour / 24)
    q = 420 + 12 * outdoor_temp + 0.4 * np.arange(n)
    q += np.where((hour >= 8) & (hour < 18), 35, -15)

    return pd.DataFrame(
        {
            "q_delivered_kw": q.astype(float),
            "is_steady": [False] * 8 + [True] * (n - 8),
            "outdoor_temp": outdoor_temp.astype(float),
            "outdoor_rh": outdoor_rh.astype(float),
            "total_power": (q * 0.65).astype(float),
        },
        index=index,
    )


def test_q_demand_features_keep_full_frame_and_build_lags() -> None:
    addon = HVACAddon()
    df = _make_training_frame(140)

    feat_df = addon.q_demand.build_features(df)

    assert not feat_df.empty
    assert len(feat_df) == len(df)
    assert feat_df["is_business_hour"].isin([0, 1]).all()
    assert "wet_bulb" in feat_df.columns
    assert "q_lag_4" in feat_df.columns
    assert "q_lag_16" in feat_df.columns
    assert "q_lag_96" in feat_df.columns
    assert "is_steady" in feat_df.columns
    assert feat_df["is_steady"].any()


def test_pinball_loss_zero_when_predictions_match() -> None:
    y_true = pd.Series([10.0, 20.0, 30.0])
    y_pred = np.array([10.0, 20.0, 30.0])
    assert pinball_loss(y_true, y_pred, alpha=0.9) == 0.0


def test_train_q_demand_model_returns_expected_metrics() -> None:
    pytest.importorskip("lightgbm")
    pytest.importorskip("optuna")

    addon = HVACAddon()
    df = _make_training_frame()

    result = train_q_demand_model(df, addon, n_trials=2, cv_folds=3)

    assert 0.0 <= result.coverage <= 1.0
    assert result.pinball_holdout >= 0.0
    assert result.baseline_pinball_holdout >= 0.0
    assert isinstance(result.pinball_improvement, float)
    assert isinstance(result.best_params, dict)
    assert "outdoor_temp" in result.feature_importance
    assert len(result.top_features) <= 3
    assert set(result.validation) == {
        "coverage_ok",
        "pinball_improvement_ok",
        "weather_driver_ok",
    }


def test_train_q_demand_model_survives_shap_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("lightgbm")
    pytest.importorskip("optuna")

    class _FailingExplainer:
        def shap_values(self, sample):
            raise ValueError("shap exploded")

    class _FailingShap:
        def TreeExplainer(self, model):
            return _FailingExplainer()

    monkeypatch.setattr(qd_trainer, "_try_import_shap", lambda: _FailingShap())

    addon = HVACAddon()
    df = _make_training_frame()

    result = train_q_demand_model(df, addon, n_trials=2, cv_folds=3)

    assert result.model is not None
    assert isinstance(result.top_features, list)
    assert result.top_features


def test_train_and_evaluate_survives_shap_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("optuna")

    from engine.core.pipeline import run_pipeline
    from engine.core.ml.trainer import train_and_evaluate

    class _FailingExplainer:
        def shap_values(self, sample):
            raise ValueError("shap exploded")

    class _FailingShap:
        def TreeExplainer(self, model):
            return _FailingExplainer()

    monkeypatch.setattr(power_trainer, "_try_import_shap", lambda: _FailingShap())

    addon = HVACAddon()
    index = pd.date_range("2024-01-01", periods=220, freq="15min")
    hour = index.hour.to_numpy()
    df = pd.DataFrame(
        {
            "total_power": (420 + 3 * np.sin(2 * np.pi * hour / 24) + np.arange(220) * 0.5).astype(float),
            "temp": (28 + 2 * np.sin(2 * np.pi * hour / 24)).astype(float),
            "rh": (65 + 5 * np.cos(2 * np.pi * hour / 24)).astype(float),
        },
        index=index,
    )

    result = train_and_evaluate(df, addon, target="total_power", n_trials=2)

    assert result.model is not None
    assert result.shap_vals is None
