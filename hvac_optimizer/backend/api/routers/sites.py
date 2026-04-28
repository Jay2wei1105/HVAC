from fastapi import APIRouter, HTTPException, Query
from hvac_optimizer.backend.api.schemas import SiteCreate, SiteResponse, ActivateImportHistoryRequest
from hvac_optimizer.backend.api.store import ACTIVE_DATASETS
from hvac_optimizer.backend.api.routers.data import save_metadata
from hvac_optimizer.backend.services.history_service import (
    apply_history_snapshot,
    ensure_import_history_backfill,
    history_summary_label,
)
import uuid

router = APIRouter()

DRAFT_SENTINEL = "__draft_current__"


def _is_analysis_ready(info: dict) -> bool:
    """True when equipment + ML training succeeded (dashboard / optimization may run)."""
    ml = info.get("ml_results")
    if not ml or ml.get("status") != "success":
        return False
    return info.get("equipment") is not None


def _dropdown_label(info: dict, sid: str) -> str:
    if info.get("project_display_name"):
        return str(info["project_display_name"])
    hist = info.get("import_history") or []
    if hist:
        return history_summary_label(hist[-1])
    return sid


@router.get("/list")
async def list_sites(completed_only: bool = Query(False, description="Only trained, analysis-ready projects")):
    """List sites; optional filter for completed imports (Streamlit dashboard / optimization picker)."""
    out: list[dict] = []
    for sid, info in ACTIVE_DATASETS.items():
        completed = _is_analysis_ready(info)
        if completed_only and not completed:
            continue
        out.append(
            {
                "site_id": sid,
                "dataset_id": info.get("dataset_id"),
                "imported": info.get("equipment") is not None,
                "completed": completed,
                "project_display_name": info.get("project_display_name"),
                "label": _dropdown_label(info, sid),
            }
        )
    return out


@router.get("/{site_id}/import-history")
async def get_import_history(site_id: str):
    """
    List completed import snapshots (time range, row count) for the navbar dropdown.

    Includes ``show_draft_option`` when the workspace has a new upload not yet archived.
    """
    if site_id not in ACTIVE_DATASETS:
        raise HTTPException(status_code=404, detail="Unknown site")
    b = ACTIVE_DATASETS[site_id]
    if ensure_import_history_backfill(b):
        save_metadata()
    raw_items = list(b.get("import_history") or [])
    items = sorted(raw_items, key=lambda x: x.get("completed_at") or "", reverse=True)
    hist_paths = {h.get("cleaned_path") for h in items if h.get("cleaned_path")}
    cur = b.get("cleaned_path")
    show_draft = bool(cur and cur not in hist_paths and len(items) > 0)
    out = []
    for rec in items:
        out.append(
            {
                "history_id": rec["history_id"],
                "label": history_summary_label(rec),
                "range_start": rec.get("range_start"),
                "range_end": rec.get("range_end"),
                "row_count": rec.get("row_count"),
                "completed_at": rec.get("completed_at"),
                "source_filename": rec.get("source_filename"),
                "cleaned_path": rec.get("cleaned_path"),
            }
        )
    return {
        "items": out,
        "active_history_id": b.get("active_history_id"),
        "show_draft_option": show_draft,
        "draft_sentinel": DRAFT_SENTINEL,
    }


@router.post("/{site_id}/import-history/activate")
async def activate_import_history(site_id: str, body: ActivateImportHistoryRequest):
    """Restore cleaned file path, mapping, equipment, and ML results from a snapshot."""
    if site_id not in ACTIVE_DATASETS:
        raise HTTPException(status_code=404, detail="Unknown site")
    if body.history_id == DRAFT_SENTINEL:
        raise HTTPException(status_code=400, detail="Cannot activate draft sentinel via API")
    b = ACTIVE_DATASETS[site_id]
    try:
        apply_history_snapshot(b, body.history_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="History record not found")
    save_metadata()
    return {"status": "success", "active_history_id": body.history_id}


@router.post("/", response_model=SiteResponse)
async def create_site(site: SiteCreate):
    return SiteResponse(
        site_id=f"site_{uuid.uuid4().hex[:8]}",
        **site.model_dump()
    )
