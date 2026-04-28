"""Import history snapshots: time-range labels and persistence helpers."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def _timestamp_source_column(mappings: list[dict[str, Any]] | None, columns: list[str]) -> str | None:
    """Resolve which raw column holds timestamps."""
    if mappings:
        for m in mappings:
            if m.get("target") == "timestamp" and m.get("source"):
                return str(m["source"])
    for c in columns:
        cl = str(c).lower()
        if cl in ("ts", "timestamp", "time", "datetime", "date"):
            return str(c)
    return None


def compute_time_range(
    cleaned_path: str,
    mappings: list[dict[str, Any]] | None,
    columns: list[str],
) -> tuple[str | None, str | None]:
    """
    Return ISO8601 bounds for the timestamp column in the cleaned CSV.

    Returns (None, None) if no usable time column or parse failure.
    """
    ts_col = _timestamp_source_column(mappings, columns)
    if not ts_col:
        return None, None
    try:
        path = Path(cleaned_path)
        if not path.is_file():
            return None, None
        df = pd.read_csv(path, usecols=[ts_col], low_memory=False)
        if ts_col not in df.columns:
            return None, None
        series = pd.to_datetime(df[ts_col], errors="coerce").dropna()
        if series.empty:
            return None, None
        mn, mx = series.min().to_pydatetime(), series.max().to_pydatetime()
        return mn.isoformat(), mx.isoformat()
    except Exception:
        return None, None


def build_history_record(site_bundle: dict[str, Any]) -> dict[str, Any]:
    """
    Build one import_history entry from a full site dataset bundle.

    Expects keys: dataset_id, cleaned_path, original_path, columns, mapping,
    equipment, ml_results, cleaning_stats.
    """
    history_id = f"ih_{uuid.uuid4().hex[:12]}"
    cols = site_bundle.get("columns") or []
    mapping = site_bundle.get("mapping")
    cleaned = site_bundle.get("cleaned_path") or ""
    rs, re = compute_time_range(str(cleaned), mapping, list(cols))
    orig = site_bundle.get("original_path") or ""
    source_filename = Path(str(orig)).name if orig else ""

    return {
        "history_id": history_id,
        "dataset_id": site_bundle.get("dataset_id"),
        "cleaned_path": cleaned,
        "original_path": orig,
        "source_filename": source_filename,
        "range_start": rs,
        "range_end": re,
        "row_count": (site_bundle.get("cleaning_stats") or {}).get("final_rows"),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "mapping": mapping,
        "equipment": site_bundle.get("equipment"),
        "ml_results": site_bundle.get("ml_results"),
        "cleaning_stats": site_bundle.get("cleaning_stats"),
        "columns": list(cols) if cols else [],
    }


def history_summary_label(rec: dict[str, Any]) -> str:
    """Human-readable label for dropdowns (data interval + rows)."""
    rs, re = rec.get("range_start"), rec.get("range_end")
    rows = rec.get("row_count")
    fn = rec.get("source_filename") or ""
    row_part = f"{rows:,} 筆" if isinstance(rows, int) else f"{rows or '?'} 筆"
    if rs and re:
        rs_d = str(rs)[:10]
        re_d = str(re)[:10]
        return f"{rs_d} ~ {re_d} · {row_part}"
    if fn:
        return f"{fn} · {row_part}"
    hid = rec.get("history_id") or ""
    return f"{hid[:10]}… · {row_part}"


def append_completed_import(site_bundle: dict[str, Any]) -> dict[str, Any]:
    """
    Append a snapshot to site_bundle['import_history'] and set active_history_id.

    Mutates site_bundle in place; does not persist to disk.
    """
    rec = build_history_record(site_bundle)
    hist: list[dict[str, Any]] = site_bundle.setdefault("import_history", [])
    hist.append(rec)
    site_bundle["active_history_id"] = rec["history_id"]
    return rec


def apply_history_snapshot(site_bundle: dict[str, Any], history_id: str) -> None:
    """Restore dataset fields from a history record. Mutates site_bundle."""
    hist: list[dict[str, Any]] = site_bundle.get("import_history") or []
    rec = next((h for h in hist if h.get("history_id") == history_id), None)
    if not rec:
        raise KeyError(f"Unknown history_id: {history_id}")
    site_bundle["dataset_id"] = rec.get("dataset_id")
    site_bundle["cleaned_path"] = rec.get("cleaned_path")
    site_bundle["original_path"] = rec.get("original_path")
    site_bundle["mapping"] = rec.get("mapping")
    site_bundle["equipment"] = rec.get("equipment")
    site_bundle["ml_results"] = rec.get("ml_results")
    site_bundle["cleaning_stats"] = rec.get("cleaning_stats")
    site_bundle["columns"] = rec.get("columns") or []
    site_bundle["active_history_id"] = history_id


def ensure_import_history_backfill(site_bundle: dict[str, Any]) -> bool:
    """
    If site has completed equipment but empty import_history, add one entry.

    Returns True if metadata should be saved.
    """
    hist = site_bundle.get("import_history")
    if hist is None:
        site_bundle["import_history"] = []
    if site_bundle["import_history"]:
        return False
    if not site_bundle.get("equipment") or not site_bundle.get("cleaned_path"):
        return False
    rec = build_history_record(site_bundle)
    site_bundle["import_history"] = [rec]
    site_bundle["active_history_id"] = rec["history_id"]
    return True
