"""
HVAC physics computation kernel — used by OptimizationService.

All functions are pure (no I/O) and return unrounded values where used
as optimizer objectives to preserve numerical gradients.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# ─── Physics constants ─────────────────────────────────────────────────────────
CHWS_COP   = 0.025   # +2.5 % COP per +1 °C CHWS raise (evaporator side)
CWS_COP    = 0.030   # +3.0 % COP per -1 °C CWS drop   (condenser side)
CT_HZ_CWS  = 0.06    # approx °C CWS drop per +1 Hz CT fan increase
COP_CLIP   = (0.65, 1.50)
NT_PER_KWH = 3.85    # NT$/kWh
OP_HOURS   = 12.0    # assumed daily operating hours
DAYS_YR    = 365


# ─── helpers ──────────────────────────────────────────────────────────────────

def safe_mean(series: pd.Series, default: float = 0.0) -> float:
    vals = series.dropna()
    return float(vals.mean()) if len(vals) > 0 else default


def baseline(df: pd.DataFrame, cols: dict[str, str | None]) -> dict[str, float]:
    """Extract real operating-period statistics from a cleaned DataFrame."""
    work = df.copy()
    rename = {v: k for k, v in cols.items() if v and v in work.columns and k != "ts"}
    work = work.rename(columns=rename)
    has = lambda c: c in work.columns  # noqa: E731

    interval_h = 5 / 60
    if has("ts"):
        ts = pd.to_datetime(work["ts"], errors="coerce").dropna()
        if len(ts) > 1:
            interval_h = (ts.max() - ts.min()).total_seconds() / 3600 / (len(ts) - 1)

    op_mask = work["total_kw"] > 30 if has("total_kw") else pd.Series([True] * len(work))

    def _op(col: str, default: float = 0.0) -> float:
        return safe_mean(work.loc[op_mask, col], default) if has(col) else default

    total_kw    = safe_mean(work["total_kw"]) if has("total_kw") else 0.0
    ch_kw       = _op("ch_kw")
    chwp_kw     = _op("chwp_kw")
    cwp_kw      = _op("cwp_kw")
    ct_kw       = _op("ct_kw")
    chws_temp   = _op("chw_supply",  7.0)
    cws_temp    = _op("cw_supply",  27.0)
    chwp_hz     = _op("chwp_hz",   45.0)
    ct_hz       = _op("ct_hz",     42.0)
    oa_temp_avg = safe_mean(work["oa_temp"], 30.0) if has("oa_temp") else 30.0

    return {
        "total_kw":    round(total_kw, 2),
        "ch_kw":       round(ch_kw, 2),
        "chwp_kw":     round(chwp_kw, 2),
        "cwp_kw":      round(cwp_kw, 2),
        "ct_kw":       round(ct_kw, 2),
        "chws_temp":   round(chws_temp, 2),
        "cws_temp":    round(cws_temp, 2),
        "chwp_hz":     round(chwp_hz, 1),
        "ct_hz":       round(ct_hz, 1),
        "oa_temp_avg": round(oa_temp_avg, 1),
        "total_kwh":   round(total_kw * len(work) * interval_h, 0),
        "cost_daily":  round(total_kw * OP_HOURS * NT_PER_KWH, 0),
    }


def raw_savings(x: np.ndarray, b: dict[str, float]) -> float:
    """
    Unrounded total kW savings for a setpoint vector x = [chws, chwp, ct, cws].

    Gradient-safe: no rounding, no integer operations.
    """
    chws_new, chwp_new, ct_new, cws_new = float(x[0]), float(x[1]), float(x[2]), float(x[3])
    chws_base = b["chws_temp"] or 7.0
    cws_base  = b["cws_temp"]  or 27.0
    chwp_base = max(b["chwp_hz"], 10.0)
    ct_base   = max(b["ct_hz"],   10.0)
    oa_avg    = b["oa_temp_avg"]

    r_chws       = np.clip(1 + CHWS_COP * (chws_new - chws_base), *COP_CLIP)
    ch_sv_chws   = b["ch_kw"] * (1 - 1 / r_chws)
    cws_achieved = max(cws_new - (ct_new - ct_base) * CT_HZ_CWS, max(oa_avg - 8.0, 22.0))
    r_cws        = np.clip(1 + CWS_COP * (cws_base - cws_achieved), *COP_CLIP)
    ch_sv_cws    = b["ch_kw"] * (1 - 1 / r_cws)
    total_ch     = min(ch_sv_chws + ch_sv_cws, b["ch_kw"] * 0.42)
    chwp_sv      = b["chwp_kw"] * (1 - (chwp_new / chwp_base) ** 3)
    ct_sv        = b["ct_kw"]   * (1 - (ct_new   / ct_base)   ** 3)
    return total_ch + chwp_sv + ct_sv


def compute_core(b: dict[str, float], params: dict[str, float]) -> dict[str, Any]:
    """Formatted physics result for a single setpoint combination."""
    chws_new = params["chws"];  chwp_new = params["chwp"]
    ct_new   = params["ct_fan"]; cws_new = params["cws"]
    chws_base = b["chws_temp"] or 7.0;   cws_base  = b["cws_temp"]  or 27.0
    chwp_base = max(b["chwp_hz"], 10.0); ct_base   = max(b["ct_hz"],   10.0)
    oa_avg    = b["oa_temp_avg"]

    r_chws       = np.clip(1 + CHWS_COP * (chws_new - chws_base), *COP_CLIP)
    ch_sv_chws   = b["ch_kw"] * (1 - 1 / r_chws)
    cws_achieved = max(cws_new - (ct_new - ct_base) * CT_HZ_CWS, max(oa_avg - 8.0, 22.0))
    r_cws        = np.clip(1 + CWS_COP * (b["cws_temp"] - cws_achieved), *COP_CLIP)
    ch_sv_cws    = b["ch_kw"] * (1 - 1 / r_cws)
    sv_ch   = float(min(ch_sv_chws + ch_sv_cws, b["ch_kw"] * 0.42))
    sv_chwp = float(b["chwp_kw"] * (1 - (chwp_new / chwp_base) ** 3))
    sv_ct   = float(b["ct_kw"]   * (1 - (ct_new   / ct_base)   ** 3))
    sv_cwp  = 0.0

    total_sv  = sv_ch + sv_chwp + sv_ct + sv_cwp
    total_opt = max(0.0, b["total_kw"] - total_sv)
    pct       = total_sv / b["total_kw"] * 100 if b["total_kw"] > 0 else 0.0

    return {
        "optimized": {
            "total_kw":   round(total_opt, 1),
            "ch_kw":      round(max(0, b["ch_kw"]   - sv_ch),   1),
            "chwp_kw":    round(max(0, b["chwp_kw"] - sv_chwp), 1),
            "cwp_kw":     round(b["cwp_kw"],  1),
            "ct_kw":      round(max(0, b["ct_kw"]   - sv_ct),   1),
            "cost_daily": round(total_opt * OP_HOURS * NT_PER_KWH, 0),
        },
        "savings": {
            "total_kw":    round(total_sv, 1),
            "total_pct":   round(pct, 1),
            "ch_kw":       round(sv_ch, 1),
            "chwp_kw":     round(sv_chwp, 1),
            "ct_kw":       round(sv_ct, 1),
            "cwp_kw":      round(sv_cwp, 1),
            "cost_daily":  round(total_sv * OP_HOURS * NT_PER_KWH, 0),
            "cost_annual": round(total_sv * OP_HOURS * NT_PER_KWH * DAYS_YR, 0),
        },
    }


def ml_baseline_check(df: pd.DataFrame, mapping: list, model_path: str) -> float | None:
    """Load XGBoost model and predict mean total_kw on actual data (validation only)."""
    try:
        import joblib
        from hvac_optimizer.backend.services.ml_service import MLService
        model  = joblib.load(model_path)
        X      = MLService._construct_features(df, mapping)
        return float(model.predict(X).mean()) if not X.empty else None
    except Exception:
        return None
