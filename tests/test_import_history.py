"""Tests for HVAC import history (time range + snapshots)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from hvac_optimizer.backend.services.history_service import (
    append_completed_import,
    apply_history_snapshot,
    build_history_record,
    compute_time_range,
    history_summary_label,
)
from hvac_optimizer.backend.services.site_migration import project_display_name_from_range


def test_compute_time_range_with_mapping() -> None:
    path = Path("data/history/_test_import_history/clean.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2024-06-01", periods=8, freq="h"),
            "Total_kW": range(8),
        }
    )
    df.to_csv(path, index=False)
    rs, re = compute_time_range(
        str(path),
        [{"source": "ts", "target": "timestamp"}],
        ["ts", "Total_kW"],
    )
    assert rs is not None and re is not None
    assert rs <= re
    shutil.rmtree(path.parent)


def test_build_and_apply_roundtrip() -> None:
    bundle = {
        "dataset_id": "ds_abc",
        "cleaned_path": "/tmp/cleaned.csv",
        "original_path": "/tmp/raw.csv",
        "columns": ["ts"],
        "mapping": [{"source": "ts", "target": "timestamp"}],
        "equipment": {"chillers": []},
        "ml_results": {"r2_score": 0.9},
        "cleaning_stats": {"final_rows": 100},
    }
    rec = build_history_record(bundle)
    assert rec["history_id"].startswith("ih_")
    bundle["import_history"] = []
    append_completed_import(bundle)
    assert len(bundle["import_history"]) == 1
    hid = bundle["import_history"][0]["history_id"]
    assert bundle["active_history_id"] == hid

    bundle["dataset_id"] = "ds_changed"
    bundle["ml_results"] = None
    apply_history_snapshot(bundle, hid)
    assert bundle["dataset_id"] == "ds_abc"
    assert bundle["ml_results"] == {"r2_score": 0.9}


def test_project_display_name_from_range_same_year() -> None:
    assert (
        project_display_name_from_range("2024-07-11T00:00:00", "2024-08-11T23:59:59")
        == "2024-0711-0811"
    )


def test_project_display_name_from_range_cross_year() -> None:
    assert project_display_name_from_range(
        "2024-12-15T00:00:00", "2025-01-15T00:00:00"
    ) == "2024-1215-2025-0115"


def test_history_summary_label() -> None:
    lbl = history_summary_label(
        {
            "history_id": "ih_x",
            "range_start": "2024-01-01T00:00:00",
            "range_end": "2024-01-31T00:00:00",
            "row_count": 1234,
            "source_filename": "f.csv",
        }
    )
    assert "2024-01-01" in lbl and "2024-01-31" in lbl
    assert "1,234" in lbl
