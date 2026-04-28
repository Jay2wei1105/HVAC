from __future__ import annotations

import math
import pandas as pd
from fastapi import APIRouter, HTTPException

from hvac_optimizer.backend.api.schemas import RealtimeAnalysisResponse
from hvac_optimizer.backend.api.store import ACTIVE_DATASETS
from hvac_optimizer.backend.services.analytics_service import AnalyticsService
from hvac_optimizer.backend.services.core_hvac_service import CoreHVACService

router = APIRouter()


def _json_safe(value):
    """
    Convert NaN/Inf recursively so FastAPI JSON serialization won't fail.
    """
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_json_safe(v) for v in value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


@router.get("/realtime", response_model=RealtimeAnalysisResponse)
async def get_realtime_data(site_id: str):
    if site_id not in ACTIVE_DATASETS:
        return RealtimeAnalysisResponse(
            kpi={
                "month_power": {"value": "--", "unit": "kWh", "trend": "flat", "pct": "0%"},
                "month_cost": {"value": "--", "unit": "NT$", "trend": "flat", "pct": "0%"},
                "contract_capacity": {"value": "800", "unit": "kW", "usage": "0%"},
                "avg_load": {"value": "--", "unit": "kW", "trend": "flat", "pct": "0%"},
            },
            equipment={"chiller": "Not set", "cooling_tower": "Not set"},
            charts={"timeseries": "empty", "heatmap": "empty"},
        )

    ds = ACTIVE_DATASETS[site_id]
    cleaned_parquet = ds.get("cleaned_parquet_path")
    if cleaned_parquet:
        df = pd.read_parquet(cleaned_parquet)
    elif ds.get("cleaned_path"):
        df = pd.read_csv(ds["cleaned_path"])
    else:
        raise HTTPException(status_code=404, detail="No cleaned dataset available")

    power_col = "total_power" if "total_power" in df.columns else next(
        (c for c in df.columns if "power" in c.lower() or "kw" in c.lower()),
        None,
    )
    if power_col is None:
        return RealtimeAnalysisResponse(
            kpi={
                "month_power": {"value": "0", "unit": "kWh", "trend": "flat", "pct": "0%"},
                "month_cost": {"value": "0", "unit": "NT$", "trend": "flat", "pct": "0%"},
                "contract_capacity": {"value": "800", "unit": "kW", "usage": "N/A"},
                "avg_load": {"value": str(len(df)), "unit": "rows", "trend": "flat", "pct": "0%"},
            },
            equipment={"chiller": "Unknown", "cooling_tower": "Unknown"},
            charts={"timeseries": "none", "heatmap": "none"},
        )

    total_kwh = float(df[power_col].sum())
    avg_kw = float(df[power_col].mean())
    usage_pct = avg_kw / 800 * 100
    return RealtimeAnalysisResponse(
        kpi={
            "month_power": {"value": f"{total_kwh:,.0f}", "unit": "kWh", "trend": "flat", "pct": "N/A"},
            "month_cost": {"value": f"{total_kwh * 3.85:,.0f}", "unit": "NT$", "trend": "flat", "pct": "N/A"},
            "contract_capacity": {"value": "800", "unit": "kW", "usage": f"{usage_pct:.1f}%"},
            "avg_load": {"value": f"{avg_kw:.1f}", "unit": "kW", "trend": "flat", "pct": "N/A"},
        },
        equipment={"chiller": "Configured", "cooling_tower": "Configured"},
        charts={"timeseries": "real_data", "heatmap": "real_data"},
    )


@router.get("/analytics")
async def get_analytics(site_id: str):
    if site_id not in ACTIVE_DATASETS:
        raise HTTPException(status_code=404, detail="No dataset for this site")
    ds = ACTIVE_DATASETS[site_id]
    cleaned_parquet = ds.get("cleaned_parquet_path")
    if cleaned_parquet:
        try:
            df = pd.read_parquet(cleaned_parquet).reset_index().rename(columns={"index": "timestamp"})
        except FileNotFoundError:
            cleaned_csv = ds.get("cleaned_path")
            if cleaned_csv:
                df = pd.read_csv(cleaned_csv)
            else:
                raise HTTPException(
                    status_code=404,
                    detail="Cleaned dataset file not found. Please re-import or retrain this project.",
                )
    elif ds.get("cleaned_path"):
        df = pd.read_csv(ds["cleaned_path"])
    else:
        raise HTTPException(status_code=404, detail="No cleaned file found; please complete onboarding first")

    mappings = ds.get("mapping") or []
    analytics = AnalyticsService.compute(df, mappings)
    analytics["quality_report"] = ds.get("cleaning_stats")
    analytics["ml_results"] = ds.get("ml_results")
    analytics["optimization_results"] = ds.get("optimization_results")
    analytics["mpc_summary"] = ds.get("mpc_summary")
    analytics["report_summary"] = ds.get("report_summary")
    analytics["dashboard_payload"] = ds.get("dashboard_payload")
    analytics["equipment"] = ds.get("equipment")
    analytics["site_meta"] = {
        "site_id": site_id,
        "dataset_id": ds.get("dataset_id"),
        "source_filename": ds.get("source_filename") or ds.get("original_path"),
    }
    return _json_safe(analytics)


@router.post("/optimize")
async def run_optimization(site_id: str, params: dict):
    if site_id not in ACTIVE_DATASETS:
        raise HTTPException(status_code=404, detail="No dataset for this site")
    ds = ACTIVE_DATASETS[site_id]
    if not ds.get("cleaned_parquet_path") or not (ds.get("ml_results") or {}).get("model_bundle_path"):
        raise HTTPException(status_code=422, detail="Training artifacts are missing; please complete onboarding first")

    first = next(iter(params.values()), None)
    if not isinstance(first, list):
        params = {k: [float(v), float(v)] for k, v in params.items()}

    result = CoreHVACService.optimize_site(
        site_id=site_id,
        dataset_id=ds["dataset_id"],
        cleaned_parquet_path=ds["cleaned_parquet_path"],
        mappings=ds.get("mapping") or [],
        model_bundle_path=ds["ml_results"]["model_bundle_path"],
        bounds=params,
        quality_report=ds.get("cleaning_stats"),
        ml_results=ds.get("ml_results"),
    )
    ds["optimization_results"] = result
    ds["mpc_summary"] = result.get("mpc")
    ds["report_summary"] = result.get("artifacts")
    ds["dashboard_payload"] = result.get("dashboard_payload")

    from hvac_optimizer.backend.api.routers.data import save_metadata

    save_metadata()
    return result
