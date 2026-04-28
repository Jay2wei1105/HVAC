"""
HVAC Optimization Service — bound-constrained setpoint optimizer.

Accepts [lo, hi] bounds for each HVAC setpoint and uses scipy L-BFGS-B
to find the combination that maximises total kW savings.  Physics kernels
live in opt_physics.py; this module owns only I/O and orchestration.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

from hvac_optimizer.backend.services.analytics_service import AnalyticsService
from hvac_optimizer.backend.services.opt_physics import (
    baseline, compute_core, ml_baseline_check, raw_savings,
)


def _load_base(ds: dict) -> tuple[pd.DataFrame, list, dict]:
    """Load cleaned CSV and extract baseline stats. Raises ValueError on missing data."""
    cleaned = ds.get("cleaned_path") or ""
    if not cleaned or not os.path.exists(cleaned):
        raise ValueError("No cleaned dataset — please complete onboarding first.")
    df      = pd.read_csv(cleaned)
    mapping = ds.get("mapping") or []
    cols    = AnalyticsService.resolve_cols(df, mapping)
    base    = baseline(df, cols)
    return df, mapping, base


class OptimizationService:
    """Bound-constrained HVAC setpoint optimizer (physics + ML validation)."""

    @staticmethod
    def optimize_bounds(
        site_id: str,
        bounds: dict[str, list[float]],
        ds: dict,
    ) -> dict[str, Any]:
        """
        Find energy-optimal setpoints within user-specified ranges.

        ``bounds`` keys: ``chws``, ``chwp``, ``ct_fan``, ``cws``.
        Each value is ``[lo, hi]`` (float).

        Returns:
          - ``baseline``       — real data operating stats
          - ``optimized``      — equipment kW at optimal setpoints
          - ``savings``        — per-equipment and total kW / NT$ savings
          - ``optimal_params`` — the setpoints found by the optimizer
          - ``bounds_used``    — resolved [lo, hi] for each parameter
          - ``positions``      — 0 = lower bound, 1 = upper bound
          - ``sensitivity``    — isolated kW contribution per parameter
          - ``converged``      — scipy convergence flag
        """
        from scipy.optimize import minimize

        try:
            df, mapping, base = _load_base(ds)
        except ValueError as exc:
            return {"error": str(exc)}

        # ── Resolve bounds (fallback to ±range around actual baseline) ────────
        b0 = base
        sp_bounds = [
            bounds.get("chws",   [b0["chws_temp"] or 7.0,  12.0]),
            bounds.get("chwp",   [25.0, b0["chwp_hz"]   or 52.0]),
            bounds.get("ct_fan", [25.0, 60.0]),
            bounds.get("cws",    [22.0, b0["cws_temp"]  or 29.0]),
        ]
        sp_bounds = [[float(min(b)), float(max(b))] for b in sp_bounds]
        chws_b, chwp_b, ct_b, cws_b = sp_bounds

        # ── L-BFGS-B optimisation (raw_savings has continuous gradients) ──────
        opt = minimize(
            lambda x: -raw_savings(x, base),
            x0=np.array([(b[0] + b[1]) / 2 for b in sp_bounds]),
            bounds=sp_bounds,
            method="L-BFGS-B",
            options={"maxiter": 300, "ftol": 1e-8},
        )

        optimal = {
            "chws":   round(float(opt.x[0]), 2),
            "chwp":   round(float(opt.x[1]), 1),
            "ct_fan": round(float(opt.x[2]), 1),
            "cws":    round(float(opt.x[3]), 2),
        }
        core = compute_core(base, optimal)

        # ── Position within each range ────────────────────────────────────────
        def _pos(v: float, lo: float, hi: float) -> float:
            return round((v - lo) / (hi - lo), 3) if hi > lo else 0.5

        positions = {
            "chws":   _pos(optimal["chws"],   *chws_b),
            "chwp":   _pos(optimal["chwp"],   *chwp_b),
            "ct_fan": _pos(optimal["ct_fan"], *ct_b),
            "cws":    _pos(optimal["cws"],    *cws_b),
        }

        # ── Per-parameter sensitivity ─────────────────────────────────────────
        base_params = {
            "chws":   base["chws_temp"] or 7.0,
            "chwp":   base["chwp_hz"]   or 45.0,
            "ct_fan": base["ct_hz"]     or 42.0,
            "cws":    base["cws_temp"]  or 27.0,
        }
        sensitivity = {
            lbl: round(float(
                compute_core(base, {**base_params, key: optimal[key]})["savings"]["total_kw"]
            ), 1)
            for key, lbl in (("chws", "CHWS"), ("chwp", "CHWP"), ("ct_fan", "CT"), ("cws", "CWS"))
        }

        # ── ML baseline validation (informational) ────────────────────────────
        ml_path    = (ds.get("ml_results") or {}).get("model_path") or ""
        ml_pred_kw = ml_baseline_check(df, mapping, ml_path) if os.path.exists(ml_path) else None

        return {
            "baseline": {
                "total_kw":   base["total_kw"],
                "ch_kw":      base["ch_kw"],
                "chwp_kw":    base["chwp_kw"],
                "cwp_kw":     base["cwp_kw"],
                "ct_kw":      base["ct_kw"],
                "cost_daily": base["cost_daily"],
                "chws_temp":  base["chws_temp"],
                "cws_temp":   base["cws_temp"],
                "chwp_hz":    base["chwp_hz"],
                "ct_hz":      base["ct_hz"],
            },
            **core,
            "optimal_params": optimal,
            "bounds_used":    {"chws": chws_b, "chwp": chwp_b, "ct_fan": ct_b, "cws": cws_b},
            "positions":      positions,
            "sensitivity":    sensitivity,
            "converged":      bool(opt.success),
            "model_used":     "ml+physics" if ml_pred_kw else "physics",
            "r2_score":       (ds.get("ml_results") or {}).get("r2_score"),
            "ml_baseline_kw": round(ml_pred_kw, 1) if ml_pred_kw else None,
        }

    @staticmethod
    def run(site_id: str, params: dict, ds: dict) -> dict[str, Any]:
        """Legacy single-point evaluation (wraps optimize_bounds with zero-width ranges)."""
        bounds = {k: [float(v), float(v)] for k, v in params.items()
                  if k in ("chws", "chwp", "ct_fan", "cws")}
        return OptimizationService.optimize_bounds(site_id, bounds, ds)
