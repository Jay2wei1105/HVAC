"""Unit tests for HVAC B+ derived features (q_delivered_kw, is_steady)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from engine.addons.hvac.features import compute_hvac_features


def test_q_delivered_kw_from_measured_flow() -> None:
    """Q = flow/60 * rho * Cp * dT (kW)."""
    df = pd.DataFrame(
        {
            "chw_supply_temp": [7.0, 7.0],
            "chw_return_temp": [12.0, 14.0],
            "chw_flow_lpm": [120.0, 120.0],
            "total_power": [100.0, 110.0],
            "ch_kw": [80.0, 85.0],
        }
    )
    out = compute_hvac_features(df)
    dt = 5.0
    expected = 120.0 / 60.0 * 1.0 * 4.186 * dt
    assert np.isclose(out["q_delivered_kw"].iloc[0], expected)
    assert out["q_delivered_kw"].iloc[1] > out["q_delivered_kw"].iloc[0]


def test_q_delivered_kw_from_chwp_freq_affinity() -> None:
    """When flow column missing, use rated_lpm * (chwp_freq / rated_hz)."""
    df = pd.DataFrame(
        {
            "chw_supply_temp": [7.0],
            "chw_return_temp": [12.0],
            "chwp_freq": [50.0],
            "total_power": [200.0],
            "ch_kw": [150.0],
        }
    )
    out = compute_hvac_features(df)
    # rated 630 L/min @ 50 Hz → flow = 630, dt = 5
    assert not out["q_delivered_kw"].isna().all()
    assert out["q_delivered_kw"].iloc[0] > 100


def test_is_steady_flags_flat_power_and_dt() -> None:
    """Flat power + stable dT → steady True after warm-up window."""
    n = 20
    p = np.full(n, 300.0)
    chws = np.full(n, 7.0)
    chwr = np.full(n, 12.0)
    df = pd.DataFrame(
        {
            "total_power": p,
            "chw_supply_temp": chws,
            "chw_return_temp": chwr,
            "chwp_freq": np.full(n, 50.0),
            "ch_kw": np.full(n, 150.0),
        }
    )
    out = compute_hvac_features(df)
    assert bool(out["is_steady"].iloc[-1]) is True
    assert bool(out["is_steady"].iloc[0]) is False


def test_columns_present_without_flow_or_freq() -> None:
    """q_delivered_kw NaN but columns exist; is_steady still computed from power."""
    df = pd.DataFrame(
        {
            "chw_supply_temp": [7.0, 7.1],
            "chw_return_temp": [12.0, 12.1],
            "total_power": [300.0, 301.0],
            "ch_kw": [150.0, 151.0],
        }
    )
    out = compute_hvac_features(df)
    assert "q_delivered_kw" in out.columns
    assert out["q_delivered_kw"].isna().all()
    assert "is_steady" in out.columns
    assert "system_kw_per_rt" in out.columns
