"""
HVAC Stage-1 derived features for Stage 2/3 (Advisory).

Water-side delivered cooling ``q_delivered_kw`` and steady-state flag ``is_steady``
per B+ methodology (replaces ``cooling_load_approx`` placeholder).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine.addons.hvac.equipment import HVAC_EQUIPMENT

CP_WATER = 4.186  # kJ/kg·K
RHO_WATER = 1.0   # kg/L (≈1)


def _chwp_rated_flow_hz() -> tuple[float, float]:
    """Rated CHWP flow (L/min) and reference Hz for affinity estimate."""
    for eq in HVAC_EQUIPMENT:
        if eq.equipment_id == "chwp":
            m = eq.metadata
            return float(m.get("rated_lpm", 630)), float(m.get("rated_hz", 50))
    return 630.0, 50.0


def _chiller_rated_kw() -> float:
    for eq in HVAC_EQUIPMENT:
        if eq.equipment_id == "chiller":
            return float(eq.metadata.get("rated_kw", 250))
    return 250.0


def _q_from_flow(df: pd.DataFrame) -> pd.Series:
    """
    Water-side thermal cooling rate (kW).

    Uses measured CHW flow when present and mostly valid; else VFD affinity
    estimate: rated_lpm * (chwp_freq / rated_hz). Requires supply/return temps.
    """
    rated_lpm, rated_hz = _chwp_rated_flow_hz()
    if "chw_flow_lpm" in df.columns and df["chw_flow_lpm"].notna().mean() > 0.5:
        flow = df["chw_flow_lpm"].astype(float)
    elif "chwp_freq" in df.columns:
        flow = rated_lpm * (df["chwp_freq"].astype(float) / rated_hz)
    else:
        return pd.Series(np.nan, index=df.index, dtype=float)

    if not {"chw_return_temp", "chw_supply_temp"}.issubset(df.columns):
        return pd.Series(np.nan, index=df.index, dtype=float)

    dt = df["chw_return_temp"].astype(float) - df["chw_supply_temp"].astype(float)
    # Q (kW) = flow (L/min) / 60 × ρ × Cp × ΔT
    return flow / 60.0 * RHO_WATER * CP_WATER * dt


def _is_steady(df: pd.DataFrame, window: int = 8) -> pd.Series:
    """
    Steady-state flag: past ``window`` rows (default 8×15min ≈ 2 h) total power
    CV < 5% and |ΔT| range < 0.5 °C when CHW temps exist.
    """
    if "total_power" not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool)

    p = df["total_power"].astype(float)
    p_mean = p.rolling(window).mean()
    p_cv = p.rolling(window).std() / p_mean.replace(0, pd.NA)

    if {"chw_return_temp", "chw_supply_temp"}.issubset(df.columns):
        dt = df["chw_return_temp"].astype(float) - df["chw_supply_temp"].astype(float)
        dt_range = dt.rolling(window).max() - dt.rolling(window).min()
        steady = (p_cv < 0.05) & (dt_range < 0.5)
    else:
        steady = p_cv < 0.05

    return steady.fillna(False).astype(bool)


def compute_hvac_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add chw_delta_t, q_delivered_kw, system_kw_per_rt, chiller_plr, is_steady.
    """
    out = df.copy()

    if {"chw_return_temp", "chw_supply_temp"}.issubset(out.columns):
        out["chw_delta_t"] = (
            out["chw_return_temp"].astype(float) - out["chw_supply_temp"].astype(float)
        )

    out["q_delivered_kw"] = _q_from_flow(out)

    if {"total_power", "q_delivered_kw"}.issubset(out.columns):
        rt = out["q_delivered_kw"].astype(float) / 3.517  # 1 RT ≈ 3.517 kW
        out["system_kw_per_rt"] = out["total_power"].astype(float) / rt.replace(0, pd.NA)

    rated = _chiller_rated_kw()
    if "ch_kw" in out.columns:
        out["chiller_plr"] = out["ch_kw"].astype(float) / rated

    out["is_steady"] = _is_steady(out)

    return out
