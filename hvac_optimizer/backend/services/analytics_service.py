"""
HVAC Analytics Service — pre-computes chart-ready payloads from cleaned DataFrames.

All returned collections are plain Python dicts/lists suitable for JSON serialisation.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


# ─── helpers ──────────────────────────────────────────────────────────────────

def _first_col(columns: list[str], patterns: list[str]) -> str | None:
    lmap = {c.lower(): c for c in columns}
    for p in patterns:
        if p.lower() in lmap:
            return lmap[p.lower()]
    return None


def _safe(v: Any) -> Any:
    if isinstance(v, (float, np.floating)):
        return None if (math.isnan(v) or math.isinf(v)) else float(round(v, 4))
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def _records(df: pd.DataFrame) -> list[dict]:
    return [{k: _safe(v) for k, v in row.items()} for row in df.to_dict("records")]


# ─── main service ─────────────────────────────────────────────────────────────

class AnalyticsService:
    """Stateless computation hub for HVAC dashboard analytics."""

    @staticmethod
    def resolve_cols(df: pd.DataFrame, mappings: list[dict]) -> dict[str, str | None]:
        """Map standard field names → actual DataFrame column names."""
        m2s: dict[str, str] = {}
        for m in (mappings or []):
            t, s = m.get("target"), m.get("source")
            # first-wins: skip if this target is already resolved
            if t and s and s in df.columns and t not in m2s:
                m2s[t] = s
        c = df.columns.tolist()
        return {
            "ts":         m2s.get("timestamp")    or _first_col(c, ["ts","timestamp","time","datetime","date"]),
            "total_kw":   m2s.get("total_power")  or _first_col(c, ["total_kw","totalkw","total_power"]),
            "ch_kw":      m2s.get("chiller_power") or _first_col(c, ["ch_kw","chillerkw","chiller_kw"]),
            "chwp_kw":    m2s.get("chwp_power")   or _first_col(c, ["chwp_kw","chwp_power"]),
            "cwp_kw":     m2s.get("cwp_power")    or _first_col(c, ["cwp_kw","cwp_power"]),
            "ct_kw":      m2s.get("ct_fan_power") or _first_col(c, ["ct_kw","ct_fan_kw","ctfankw"]),
            "chwp_hz":    m2s.get("chwp_freq")    or _first_col(c, ["chwp_hz","chwp_freq"]),
            "cwp_hz":     m2s.get("cwp_freq")     or _first_col(c, ["cwp_hz","cwp_freq"]),
            "ct_hz":      m2s.get("ct_fan_freq")  or _first_col(c, ["ct_hz","ct_fan_hz","ct_freq"]),
            "chw_supply": m2s.get("chws_temp")    or _first_col(c, ["chw_supply_temp","chws_temp","chw_supply"]),
            "chw_return": m2s.get("chwr_temp")    or _first_col(c, ["chw_return_temp","chwr_temp","chw_return"]),
            "cw_supply":  m2s.get("cws_temp")     or _first_col(c, ["cw_supply_temp","cws_temp","cw_supply"]),
            "cw_return":  m2s.get("cwr_temp")     or _first_col(c, ["cw_return_temp","cwr_temp","cw_return"]),
            "oa_temp":    m2s.get("ambient_temp") or _first_col(c, ["oa_temp","ambient_temp","outdoor_temp"]),
            "oa_rh":      m2s.get("oa_rh")        or _first_col(c, ["oa_rh","ambient_rh","humidity","rh"]),
            "chw_flow":   m2s.get("chw_flow")     or _first_col(c, ["chw_flow","chw_flow_lpm","chilled_flow","flow"]),
        }

    @staticmethod
    def compute(df: pd.DataFrame, mappings: list[dict]) -> dict[str, Any]:  # noqa: C901
        """Cleaned DataFrame + mapping list → full analytics JSON payload."""
        cols = AnalyticsService.resolve_cols(df, mappings)
        work = df.copy()

        ts_col = cols["ts"]
        if not ts_col:
            return {"error": "No timestamp column found"}

        work["_ts"] = pd.to_datetime(work[ts_col], errors="coerce")
        work = work.dropna(subset=["_ts"]).set_index("_ts")
        work.index = work.index.tz_localize(None)

        rename = {v: k for k, v in cols.items() if v and v in work.columns and k != "ts"}
        work = work.rename(columns=rename)

        has = lambda col: col in work.columns  # noqa: E731

        # ── derived metrics ────────────────────────────────────────────────────
        if has("chw_return") and has("chw_supply"):
            work["delta_chw"] = (work["chw_return"] - work["chw_supply"]).clip(0, 20)
        if has("cw_return") and has("cw_supply"):
            work["delta_cw"] = (work["cw_return"] - work["cw_supply"]).clip(-5, 15)
        if has("ch_kw") and has("delta_chw") and has("chw_flow"):
            q_kw = work["chw_flow"] * 1.163 * work["delta_chw"]
            mask = work["ch_kw"] > 0.5
            work["cop_est"] = np.nan
            work.loc[mask, "cop_est"] = (q_kw[mask] / work.loc[mask, "ch_kw"]).clip(0, 15)
        if has("total_kw"):
            peak = float(work["total_kw"].max())
            work["load_pct"] = (work["total_kw"] / peak * 100).clip(0, 100) if peak > 0 else 0.0

        work["hour"]    = work.index.hour
        work["weekday"] = work.index.weekday
        work["month"]   = work.index.month

        meta   = {"hour","weekday","month"}
        num_c  = [c for c in work.select_dtypes(include="number").columns if c not in meta]

        # ── interval estimation (hours per sample row) ─────────────────────────
        n = len(work)
        dt_h = float((work.index.max() - work.index.min()).total_seconds() / 3600) if n > 1 else 1.0
        interval_h = dt_h / (n - 1) if n > 1 else (5 / 60)

        # ── hourly series ──────────────────────────────────────────────────────
        hourly_df = work[num_c].resample("1h").mean().reset_index()
        hourly_df.rename(columns={"_ts": "ts"}, inplace=True)
        hourly_df["ts"] = hourly_df["ts"].astype(str)
        hourly_records = _records(hourly_df)

        # ── hour-of-day profile ────────────────────────────────────────────────
        prof_cols = [p for p in ["total_kw","ch_kw","chwp_kw","cwp_kw","ct_kw","oa_temp","cop_est","load_pct"] if has(p)]
        parts: list[pd.DataFrame] = []
        for pc in prof_cols:
            g = work.groupby("hour")[pc]
            parts.append(pd.DataFrame({
                f"{pc}_mean": g.mean(),
                f"{pc}_p10":  g.quantile(0.10),
                f"{pc}_p90":  g.quantile(0.90),
            }))
        profile_df   = pd.concat(parts, axis=1).reset_index() if parts else pd.DataFrame({"hour": range(24)})
        profile_recs = _records(profile_df)

        # ── weekday averages ───────────────────────────────────────────────────
        wd_recs: list[dict] = []
        if has("total_kw"):
            wd = work.groupby("weekday")["total_kw"].agg(avg_kw="mean", peak_kw="max").reset_index()
            wd_recs = _records(wd)

        # ── heatmap: hour × weekday ────────────────────────────────────────────
        heatmap: dict[str, Any] = {}
        if has("total_kw"):
            piv = work.pivot_table("total_kw", index="hour", columns="weekday", aggfunc="mean")
            piv = piv.reindex(columns=range(7), fill_value=0).fillna(0)
            heatmap = {
                "z": [[round(float(v), 1) for v in row] for row in piv.values.tolist()],
                "y": list(range(24)),
                "x": ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
            }

        # ── summary statistics ─────────────────────────────────────────────────
        stats: dict[str, Any] = {}
        if has("total_kw"):
            s = work["total_kw"]
            stats.update({
                "total_kwh":    round(float(s.sum()) * interval_h, 0),
                "peak_kw":      round(float(s.max()), 1),
                "avg_kw":       round(float(s.mean()), 1),
                "p95_kw":       round(float(s.quantile(0.95)), 1),
            })
            stats["cost_est_ntd"] = round(stats["total_kwh"] * 3.85, 0)
        if has("cop_est"):
            vc = work["cop_est"].dropna()
            if len(vc):
                stats.update({"cop_avg": round(float(vc.mean()), 2),
                               "cop_p25": round(float(vc.quantile(0.25)), 2)})
        if has("oa_temp"):
            stats.update({"oa_temp_avg": round(float(work["oa_temp"].mean()), 1),
                          "oa_temp_max": round(float(work["oa_temp"].max()), 1)})
        stats.update({
            "date_start": str(work.index.min().date()) if n else None,
            "date_end":   str(work.index.max().date()) if n else None,
            "days":       round(dt_h / 24, 1),
        })

        # ── power share ────────────────────────────────────────────────────────
        power_share = {k: round(float(work[k].mean()), 2)
                       for k in ("ch_kw","chwp_kw","cwp_kw","ct_kw") if has(k)}

        # ── load duration curve ────────────────────────────────────────────────
        ldc: list[float] = []
        if has("total_kw"):
            sv = sorted(work["total_kw"].dropna().tolist(), reverse=True)
            step = max(1, len(sv) // 600)
            ldc = [round(v, 1) for v in sv[::step]]

        # ── scatter sample (OA temp vs load) ──────────────────────────────────
        scatter: list[dict] = []
        if has("oa_temp") and has("total_kw"):
            sc_cols = ["oa_temp","total_kw","hour"] + (["ch_kw"] if has("ch_kw") else [])
            sc = work[sc_cols].dropna()
            sc = sc.sample(n=min(2000, len(sc)), random_state=42) if len(sc) > 2000 else sc
            scatter = _records(sc.reset_index().rename(columns={"_ts":"ts"}))

        # ── distributions ──────────────────────────────────────────────────────
        distrib: dict[str, Any] = {}
        for col in ["total_kw","ch_kw","oa_temp","oa_rh","cop_est","delta_chw","delta_cw"]:
            if has(col):
                vals = work[col].dropna()
                if len(vals) > 0:
                    counts, edges = np.histogram(vals, bins=40)
                    distrib[col] = {"counts": counts.tolist(),
                                    "edges":  [round(float(e), 3) for e in edges]}

        # ── monthly summary ────────────────────────────────────────────────────
        monthly: list[dict] = []
        if has("total_kw"):
            m_agg = {"total_kwh": ("total_kw", lambda x: round(float(x.sum()) * interval_h, 0)),
                     "avg_kw":    ("total_kw", "mean")}
            if has("oa_temp"):
                m_agg["avg_oa_temp"] = ("oa_temp", "mean")
            monthly = _records(work.groupby("month").agg(**m_agg).reset_index())

        return {
            "series":      {"hourly": hourly_records},
            "profile":     {"by_hour": profile_recs, "by_weekday": wd_recs, "by_month": monthly},
            "heatmap":     {"hour_vs_weekday": heatmap},
            "stats":       stats,
            "power_share": power_share,
            "scatter":     {"oa_vs_power": scatter},
            "ldc":         ldc,
            "distribution":distrib,
            "cols_found":  {k: bool(v) for k, v in cols.items()},
        }
