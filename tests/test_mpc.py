from __future__ import annotations

import pandas as pd

from engine.addons.hvac import HVACAddon
from engine.core.ml.mpc import run_mpc
from engine.core.ml.optimizer import BusinessWeights, ControlVariable


class _ConstantPowerModel:
    def predict(self, frame: pd.DataFrame):
        return [350.0] * len(frame)


class _ConstantQModel:
    def predict(self, frame: pd.DataFrame):
        return [250.0] * len(frame)


def _make_df() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=48, freq="15min")
    return pd.DataFrame(
        {
            "chw_supply_temp": [7.0] * len(idx),
            "chw_return_temp": [12.0] * len(idx),
            "chw_flow_lpm": [1000.0] * len(idx),
            "chwp_freq": [50.0] * len(idx),
            "outdoor_temp": [30.0] * len(idx),
            "outdoor_rh": [65.0] * len(idx),
            "total_power": [380.0] * len(idx),
            "ch_kw": [260.0] * len(idx),
            "chiller_count": [1.0] * len(idx),
        },
        index=idx,
    )


def test_run_mpc_returns_control_log() -> None:
    addon = HVACAddon()
    control_vars = [
        ControlVariable("chw_supply_temp", "CHWS", "C", (5.0, 10.0), (5.0, 10.0), (5.0, 10.0), 7.0),
        ControlVariable("chwp_freq", "CHWP", "Hz", (30.0, 55.0), (30.0, 55.0), (30.0, 55.0), 50.0),
    ]
    result = run_mpc(
        raw_df=_make_df(),
        addon=addon,
        power_model=_ConstantPowerModel(),
        power_feat_cols=["chw_supply_temp", "chwp_freq"],
        target="total_power",
        control_vars=control_vars,
        weights=BusinessWeights(energy=1.0, comfort=0.0, longevity=0.0),
        q_demand_model=_ConstantQModel(),
        q_demand_feat_cols=["outdoor_temp", "outdoor_rh", "q_lag_4", "q_lag_16", "q_lag_96", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_business_hour", "wet_bulb", "outdoor_temp_roll_4h"],
        horizon_steps=4,
        advisory_trials=3,
    )
    assert not result.control_log.empty
    assert "executed_action" in result.control_log.columns
    assert isinstance(result.passed_ratio_gate, bool)
