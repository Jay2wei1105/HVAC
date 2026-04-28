"""Finalize trained workspace: name project from data range, migrate storage folder, rewrite paths."""

from __future__ import annotations

import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from hvac_optimizer.backend.services.core_hvac_service import CoreHVACService


def project_display_name_from_range(range_start: str | None, range_end: str | None) -> str | None:
    """
    Build a display name like ``2024-0711-0811`` (year, start MMDD, end MMDD).

    If the end date is in a different year than the start, the end segment includes the year.
    """
    if not range_start or not range_end:
        return None
    try:
        s = datetime.fromisoformat(range_start.replace("Z", "+00:00"))
        e = datetime.fromisoformat(range_end.replace("Z", "+00:00"))
    except ValueError:
        return None
    left = f"{s.year}-{s.month:02d}{s.day:02d}"
    if e.year == s.year:
        return f"{left}-{e.month:02d}{e.day:02d}"
    return f"{left}-{e.year}-{e.month:02d}{e.day:02d}"


def allocate_unique_site_id(display_name: str, existing: set[str]) -> str:
    """Stable URL-safe site id from display name; suffix ``_2``, ``_3``… on collision."""
    slug = display_name.replace("-", "_").replace(".", "_")
    base = f"site_{slug}"
    if base not in existing:
        return base
    n = 2
    while f"{base}_{n}" in existing:
        n += 1
    return f"{base}_{n}"


def _rewrite_path_if_contains_site_segment(path_str: str, old_id: str, new_id: str) -> str:
    """Replace ``old_id`` when it appears as a single path segment."""
    p = Path(path_str)
    parts = list(p.parts)
    changed = False
    for i, seg in enumerate(parts):
        if seg == old_id:
            parts[i] = new_id
            changed = True
    if not changed:
        return path_str
    # Preserve absolute vs relative: Path concatenation from parts loses drive on some edge cases
    if path_str.startswith("/") or (len(path_str) > 2 and path_str[1] == ":"):
        return str(Path(*parts))
    return str(Path(*parts))


def deep_rewrite_site_id_in_path_strings(obj: Any, old_id: str, new_id: str) -> None:
    """Walk dict/list structures and rewrite path-like strings that contain the site folder segment."""
    if old_id == new_id:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str) and old_id in v and ("/" in v or "\\" in v):
                obj[k] = _rewrite_path_if_contains_site_segment(v, old_id, new_id)
            else:
                deep_rewrite_site_id_in_path_strings(v, old_id, new_id)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            if isinstance(item, str) and old_id in item and ("/" in item or "\\" in item):
                obj[i] = _rewrite_path_if_contains_site_segment(item, old_id, new_id)
            else:
                deep_rewrite_site_id_in_path_strings(item, old_id, new_id)


def migrate_site_directory(old_id: str, new_id: str) -> None:
    """Rename ``storage/old_id`` → ``storage/new_id`` if the source exists."""
    if old_id == new_id:
        return
    root = CoreHVACService.storage_dir()
    src = root / old_id
    dst = root / new_id
    if not src.is_dir():
        return
    if dst.exists():
        raise FileExistsError(f"Cannot migrate site folder: {dst} already exists")
    shutil.move(str(src), str(dst))


def finalize_trained_site(
    old_site_id: str,
    site_bundle: dict[str, Any],
    reserved_keys: set[str],
) -> tuple[str, str]:
    """
    Set ``project_display_name``, optionally migrate to a range-based ``site_id``.

    Args:
        old_site_id: Draft workspace key in ``ACTIVE_DATASETS``.
        site_bundle: The site's dataset bundle (mutated in place).
        reserved_keys: Existing ``ACTIVE_DATASETS`` keys **excluding** ``old_site_id``
            so the draft id does not collide with itself.

    Returns:
        ``(final_site_id, project_display_name)``.
    """
    hist = site_bundle.get("import_history") or []
    if not hist:
        raise ValueError("import_history is empty after training")

    rec = hist[-1]
    display = project_display_name_from_range(rec.get("range_start"), rec.get("range_end"))
    if not display:
        display = f"proj-{uuid.uuid4().hex[:8]}"

    new_site_id = allocate_unique_site_id(display, reserved_keys)
    site_bundle["project_display_name"] = display

    if new_site_id != old_site_id:
        migrate_site_directory(old_site_id, new_site_id)
        deep_rewrite_site_id_in_path_strings(site_bundle, old_site_id, new_site_id)

    return new_site_id, display
