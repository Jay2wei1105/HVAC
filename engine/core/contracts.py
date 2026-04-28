from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

try:
    import pandera.pandas as pa
    from pandera import Check
except ImportError:  # pragma: no cover - exercised only when dependency is missing
    pa = None
    Check = None

from .types import SensorDefinition


@dataclass
class ContractResult:
    df: pd.DataFrame
    issues: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)


def preclean_raw_frame(df: pd.DataFrame) -> ContractResult:
    cleaned = df.copy()
    original_rows = len(cleaned)

    normalized_names = [str(col).strip() for col in cleaned.columns]
    cleaned.columns = normalized_names

    unnamed_cols = [col for col in cleaned.columns if col.lower().startswith("unnamed:")]
    if unnamed_cols:
        cleaned = cleaned.drop(columns=unnamed_cols)

    empty_rows_removed = int(cleaned.isna().all(axis=1).sum())
    if empty_rows_removed:
        cleaned = cleaned.dropna(how="all")

    duplicate_rows_removed = int(cleaned.duplicated().sum())
    if duplicate_rows_removed:
        cleaned = cleaned.drop_duplicates()

    return ContractResult(
        df=cleaned,
        stats={
            "original_rows": int(original_rows),
            "final_rows": int(len(cleaned)),
            "unnamed_columns_removed": int(len(unnamed_cols)),
            "empty_rows_removed": empty_rows_removed,
            "duplicate_rows_removed": duplicate_rows_removed,
        },
    )


def coerce_and_validate_sensor_frame(
    df: pd.DataFrame,
    schema: list[SensorDefinition],
) -> ContractResult:
    cleaned = df.copy()
    coercion_failures = 0

    for sensor in schema:
        if sensor.std_name not in cleaned.columns:
            continue
        before_non_null = int(cleaned[sensor.std_name].notna().sum())
        cleaned[sensor.std_name] = pd.to_numeric(cleaned[sensor.std_name], errors="coerce")
        after_non_null = int(cleaned[sensor.std_name].notna().sum())
        coercion_failures += max(0, before_non_null - after_non_null)

    if pa is None:
        return ContractResult(
            df=cleaned,
            issues=[{"stage": "contract", "message": "pandera is not installed; schema validation skipped"}],
            stats={"numeric_coercion_failures": int(coercion_failures)},
        )

    columns: dict[str, Any] = {}
    for sensor in schema:
        if sensor.std_name not in cleaned.columns:
            continue
        checks: list[Any] = []
        if sensor.required:
            checks.append(
                Check(
                    lambda series: bool(series.notna().any()),
                    error=f"{sensor.std_name} has no usable numeric values after coercion",
                )
            )
        columns[sensor.std_name] = pa.Column(float, nullable=True, coerce=True, checks=checks)

    issues: list[dict[str, Any]] = []
    if columns:
        try:
            cleaned = pa.DataFrameSchema(columns=columns, strict=False, coerce=True).validate(cleaned, lazy=True)
        except pa.errors.SchemaErrors as exc:
            issues.extend(_format_failure_cases(exc.failure_cases, stage="contract"))

    return ContractResult(
        df=cleaned,
        issues=issues,
        stats={"numeric_coercion_failures": int(coercion_failures)},
    )


def validate_time_index(df: pd.DataFrame) -> list[dict[str, Any]]:
    if pa is None:
        return []

    schema = pa.DataFrameSchema(
        columns={},
        index=pa.Index(pa.DateTime, coerce=True),
        strict=False,
        checks=[
            Check(lambda frame: bool(frame.index.is_monotonic_increasing), error="timestamp index must be sorted"),
            Check(lambda frame: bool(frame.index.is_unique), error="timestamp index must be unique"),
        ],
    )
    try:
        schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        return _format_failure_cases(exc.failure_cases, stage="time_index")
    return []


def _format_failure_cases(failure_cases: pd.DataFrame, *, stage: str) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if failure_cases.empty:
        return issues

    for _, row in failure_cases.fillna("").iterrows():
        issues.append(
            {
                "stage": stage,
                "schema_context": str(row.get("schema_context", "")),
                "column": str(row.get("column", "")),
                "check": str(row.get("check", "")),
                "message": str(row.get("failure_case", "")),
            }
        )
    return issues
