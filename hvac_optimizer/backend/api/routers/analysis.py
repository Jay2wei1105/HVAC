from __future__ import annotations

from datetime import datetime
import math
import pandas as pd
from fastapi import APIRouter, HTTPException

from hvac_optimizer.backend.api.schemas import RealtimeAnalysisResponse
from hvac_optimizer.backend.api.store import ACTIVE_DATASETS
from hvac_optimizer.backend.services.analytics_service import AnalyticsService
from hvac_optimizer.backend.services.core_hvac_service import CoreHVACService

router = APIRouter()


def _optimization_label(snapshot: dict) -> str:
    mode = str(snapshot.get("optimization_mode") or "standard")
    mode_label = {"standard": "標準版", "fast": "加速版", "extreme": "極速版"}.get(mode, mode)
    bounds = snapshot.get("bounds_used") or {}
    parts = []
    for key in ("chws", "chwp", "cwp", "ct_fan"):
        rng = bounds.get(key)
        if isinstance(rng, list) and len(rng) == 2:
            parts.append(f"{key}:{rng[0]}-{rng[1]}")
    completed_at = str(snapshot.get("completed_at") or "")[:16].replace("T", " ")
    summary = snapshot.get("interval_summary") or {}
    saving_pct = summary.get("saving_pct")
    suffix = f" · 節能 {saving_pct}%" if saving_pct is not None else ""
    return f"{completed_at} · {mode_label} · {' / '.join(parts)}{suffix}"


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


@router.get("/optimize/status")
async def get_optimization_status(site_id: str):
    if site_id not in ACTIVE_DATASETS:
        raise HTTPException(status_code=404, detail="No dataset for this site")
    ds = ACTIVE_DATASETS[site_id]
    return _json_safe(ds.get("optimization_progress") or {"status": "idle", "percent": 0})


@router.get("/optimize/history")
async def get_optimization_history(site_id: str):
    if site_id not in ACTIVE_DATASETS:
        raise HTTPException(status_code=404, detail="No dataset for this site")
    ds = ACTIVE_DATASETS[site_id]
    items = []
    for snapshot in reversed(list(ds.get("optimization_history") or [])):
        if not isinstance(snapshot, dict):
            continue
        items.append(
            {
                "optimization_id": snapshot.get("optimization_id"),
                "label": _optimization_label(snapshot),
                "completed_at": snapshot.get("completed_at"),
                "optimization_mode": snapshot.get("optimization_mode"),
                "bounds_used": snapshot.get("bounds_used"),
                "interval_summary": snapshot.get("interval_summary"),
            }
        )
    return {
        "items": items,
        "active_optimization_id": ds.get("active_optimization_id"),
    }


@router.post("/optimize/history/activate")
async def activate_optimization_history(site_id: str, body: dict):
    if site_id not in ACTIVE_DATASETS:
        raise HTTPException(status_code=404, detail="No dataset for this site")
    optimization_id = body.get("optimization_id")
    if not optimization_id:
        raise HTTPException(status_code=400, detail="optimization_id is required")
    ds = ACTIVE_DATASETS[site_id]
    snapshots = list(ds.get("optimization_history") or [])
    match = next((item for item in snapshots if isinstance(item, dict) and item.get("optimization_id") == optimization_id), None)
    if match is None:
        raise HTTPException(status_code=404, detail="Optimization snapshot not found")
    result = match.get("result") or {}
    ds["optimization_results"] = result
    ds["active_optimization_id"] = optimization_id
    ds["mpc_summary"] = result.get("mpc")
    ds["report_summary"] = result.get("artifacts")
    ds["dashboard_payload"] = result.get("dashboard_payload")
    from hvac_optimizer.backend.api.routers.data import save_metadata

    save_metadata()
    return {"status": "success", "active_optimization_id": optimization_id}


@router.post("/optimize")
async def run_optimization(site_id: str, params: dict):
    if site_id not in ACTIVE_DATASETS:
        raise HTTPException(status_code=404, detail="No dataset for this site")
    ds = ACTIVE_DATASETS[site_id]
    if not ds.get("cleaned_parquet_path") or not (ds.get("ml_results") or {}).get("model_bundle_path"):
        raise HTTPException(status_code=422, detail="Training artifacts are missing; please complete onboarding first")

    optimization_mode = str(params.pop("optimization_mode", "standard") or "standard")
    bounds_params = {k: v for k, v in params.items()}
    first = next(iter(bounds_params.values()), None)
    if not isinstance(first, list):
        bounds_params = {k: [float(v), float(v)] for k, v in bounds_params.items()}

    ds["optimization_progress"] = {
        "status": "running",
        "percent": 1,
        "stage": "queued",
        "message": "已收到最佳化請求，準備開始執行...",
        "optimization_mode": optimization_mode,
    }

    def _update_progress(payload: dict):
        current = ds.get("optimization_progress") or {}
        current.update(payload)
        current["status"] = "running" if int(current.get("percent", 0) or 0) < 100 else "completed"
        ds["optimization_progress"] = current

    try:
        result = CoreHVACService.optimize_site(
            site_id=site_id,
            dataset_id=ds["dataset_id"],
            cleaned_parquet_path=ds["cleaned_parquet_path"],
            mappings=ds.get("mapping") or [],
            model_bundle_path=ds["ml_results"]["model_bundle_path"],
            bounds=bounds_params,
            optimization_mode=optimization_mode,
            quality_report=ds.get("cleaning_stats"),
            ml_results=ds.get("ml_results"),
            progress_callback=_update_progress,
        )
    except Exception as exc:
        ds["optimization_progress"] = {
            "status": "error",
            "percent": 100,
            "stage": "failed",
            "message": f"最佳化失敗：{exc}",
        }
        raise
    safe_result = _json_safe(result)
    ds["optimization_results"] = safe_result
    snapshot = {
        "optimization_id": f"opt_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "optimization_mode": safe_result.get("optimization_mode") or optimization_mode,
        "bounds_used": safe_result.get("bounds_used") or bounds_params,
        "interval_summary": safe_result.get("interval_summary") or {},
        "result": safe_result,
    }
    history = list(ds.get("optimization_history") or [])
    history.append(snapshot)
    ds["optimization_history"] = history
    ds["active_optimization_id"] = snapshot["optimization_id"]
    ds["mpc_summary"] = safe_result.get("mpc")
    ds["report_summary"] = safe_result.get("artifacts")
    ds["dashboard_payload"] = safe_result.get("dashboard_payload")
    ds["optimization_progress"] = {
        **(ds.get("optimization_progress") or {}),
        "status": "completed",
        "percent": 100,
        "stage": "completed",
        "message": "區間優化完成。",
    }

    from hvac_optimizer.backend.api.routers.data import save_metadata

    save_metadata()
    return safe_result
