from __future__ import annotations

import pandas as pd
import pytest

import engine.core.ml.optimizer as optimizer_module
from engine.addons.hvac import HVACAddon
from engine.core.ml.optimizer import BusinessWeights, ControlVariable, run_control_optimization


class _LinearPowerModel:
    def predict(self, frame: pd.DataFrame):
        supply = frame["chw_supply_temp"].astype(float)
        pump = frame["chwp_freq"].astype(float)
        return 700.0 - 35.0 * supply - 3.0 * pump


class _ConstantQDemandModel:
    def predict(self, frame: pd.DataFrame):
        return pd.Series([300.0] * len(frame), index=frame.index)


def _make_raw_hvac_frame() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=120, freq="15min")
    return pd.DataFrame(
        {
            "chw_supply_temp": [7.0] * len(index),
            "chw_return_temp": [12.0] * len(index),
            "chw_flow_lpm": [1000.0] * len(index),
            "chwp_freq": [50.0] * len(index),
            "outdoor_temp": [30.0] * len(index),
            "outdoor_rh": [65.0] * len(index),
            "total_power": [420.0] * len(index),
            "ch_kw": [260.0] * len(index),
            "chiller_count": [1.0] * len(index),
        },
        index=index,
    )


def test_run_control_optimization_respects_q_constraint() -> None:
    pytest.importorskip("optuna")

    addon = HVACAddon()
    raw_df = _make_raw_hvac_frame()
    model = _LinearPowerModel()
    q_model = _ConstantQDemandModel()
    control_vars = [
        ControlVariable(
            name="chw_supply_temp",
            display_name="CHWS",
            unit="C",
            l1_bounds=(5.0, 10.0),
            l2_bounds=(5.0, 10.0),
            l3_bounds=(5.0, 10.0),
            current_value=7.0,
        ),
        ControlVariable(
            name="chwp_freq",
            display_name="CHWP",
            unit="Hz",
            l1_bounds=(30.0, 55.0),
            l2_bounds=(30.0, 55.0),
            l3_bounds=(30.0, 55.0),
            current_value=50.0,
        ),
    ]

    result = run_control_optimization(
        model=model,
        feature_df=raw_df,
        feat_cols=["chw_supply_temp", "chwp_freq"],
        target="total_power",
        control_vars=control_vars,
        weights=BusinessWeights(energy=1.0, comfort=0.0, longevity=0.0),
        n_trials=30,
        addon=addon,
        q_demand_model=q_model,
        q_demand_feat_cols=["outdoor_temp", "outdoor_rh", "q_lag_4", "q_lag_16", "q_lag_96", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_business_hour", "wet_bulb", "outdoor_temp_roll_4h"],
    )

    assert result.q_required_min is not None
    assert result.q_capability is not None
    assert result.q_capability >= result.q_required_min
    assert result.feasible is True


def test_run_control_optimization_handles_empty_feature_trials(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("optuna")

    addon = HVACAddon()
    raw_df = _make_raw_hvac_frame()
    model = _LinearPowerModel()
    original_prepare = optimizer_module._prepare_prediction_frame

    def flaky_prepare(frame: pd.DataFrame, addon_obj: HVACAddon, target: str) -> pd.DataFrame:
        last_supply = float(frame["chw_supply_temp"].iloc[-1])
        if last_supply > 8.0:
            return pd.DataFrame()
        return original_prepare(frame, addon_obj, target)

    monkeypatch.setattr(optimizer_module, "_prepare_prediction_frame", flaky_prepare)

    control_vars = [
        ControlVariable(
            name="chw_supply_temp",
            display_name="CHWS",
            unit="C",
            l1_bounds=(7.5, 9.5),
            l2_bounds=(7.5, 9.5),
            l3_bounds=(7.5, 9.5),
            current_value=7.0,
        ),
        ControlVariable(
            name="chwp_freq",
            display_name="CHWP",
            unit="Hz",
            l1_bounds=(30.0, 55.0),
            l2_bounds=(30.0, 55.0),
            l3_bounds=(30.0, 55.0),
            current_value=50.0,
        ),
    ]

    result = run_control_optimization(
        model=model,
        feature_df=raw_df,
        feat_cols=["chw_supply_temp", "chwp_freq"],
        target="total_power",
        control_vars=control_vars,
        weights=BusinessWeights(energy=1.0, comfort=0.0, longevity=0.0),
        n_trials=20,
        addon=addon,
        q_demand_model=None,
        q_demand_feat_cols=[],
    )

    assert result.total_trials == 20
    assert not result.trials_df.empty
    assert "error" in result.trials_df.columns
    assert (result.trials_df["error"].fillna("").str.contains("Feature frame is empty")).any()
    assert result.predicted_power <= result.baseline_power
