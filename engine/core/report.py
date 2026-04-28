import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class QualityReport:
    """Summary of data quality assessment and cleaning results."""
    addon_id: str
    raw_rows: int
    final_rows: int
    completeness: float
    shutdown_removed: int
    spike_count: int
    flatline_count: int
    interpolated_count: int
    range_violations: List[Dict[str, Any]] = field(default_factory=list)
    cross_issues: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def missing_ratio(self) -> float:
        """Fraction of data points that are NaN after cleaning."""
        return 1.0 - self.completeness

    @property
    def anomaly_ratio(self) -> float:
        """Fraction of rows flagged as anomalies (spikes + flatlines)."""
        if self.raw_rows == 0:
            return 0.0
        return (self.spike_count + self.flatline_count) / self.raw_rows

    @property
    def should_halt(self) -> tuple[bool, str]:
        """
        PDF §5: 缺失 > 30% or outlier > 5% → 中斷，資料品質太差。
        Returns (should_halt, reason_message).
        """
        reasons: list[str] = []
        if self.missing_ratio > 0.30:
            reasons.append(
                f"缺失率 {self.missing_ratio:.1%} 超過 30% 閾值"
            )
        if self.anomaly_ratio > 0.05:
            reasons.append(
                f"異常率 {self.anomaly_ratio:.1%} 超過 5% 閾值"
            )
        if reasons:
            return True, "；".join(reasons)
        return False, ""

    @property
    def ge_html(self) -> str:
        """Returns a simplified HTML representation simulating GE Data Docs."""
        html = f"""
        <html>
            <head>
                <style>
                    body {{ font-family: sans-serif; padding: 20px; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    .success {{ color: green; font-weight: bold; }}
                    .warning {{ color: orange; font-weight: bold; }}
                    .error {{ color: red; font-weight: bold; }}
                </style>
            </head>
            <body>
                <h2>📊 Quality Report: {self.addon_id.upper()}</h2>

                <h3>Metrics</h3>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Raw Rows</td><td>{self.raw_rows}</td></tr>
                    <tr><td>Final Rows</td><td>{self.final_rows}</td></tr>
                    <tr><td>Completeness</td><td><span class="{'success' if self.completeness > 0.95 else 'warning'}">{self.completeness:.2%}</span></td></tr>
                    <tr><td>Shutdown Removed</td><td>{self.shutdown_removed}</td></tr>
                    <tr><td>Spike Count</td><td>{self.spike_count}</td></tr>
                    <tr><td>Flatline Count</td><td>{self.flatline_count}</td></tr>
                    <tr><td>Interpolated Gaps</td><td>{self.interpolated_count}</td></tr>
                    <tr><td>Range Violations</td><td><span class="{'error' if self.range_violations else 'success'}">{len(self.range_violations)}</span></td></tr>
                    <tr><td>Cross Issues</td><td><span class="{'error' if self.cross_issues else 'success'}">{len(self.cross_issues)}</span></td></tr>
                </table>
            </body>
        </html>
        """
        return html

    def summary(self) -> str:
        """Returns a human-readable summary string."""
        return (
            f"--- Quality Report: {self.addon_id.upper()} ---\n"
            f"Rows: {self.raw_rows} -> {self.final_rows}\n"
            f"Completeness: {self.completeness:.2%}\n"
            f"Shutdown Points Removed: {self.shutdown_removed}\n"
            f"Spikes Detected: {self.spike_count}\n"
            f"Flatlines Detected: {self.flatline_count}\n"
            f"Interpolated Gaps: {self.interpolated_count}\n"
            f"Range Violations: {len(self.range_violations)}\n"
            f"Cross Validation Issues: {len(self.cross_issues)}\n"
        )


def build(
    df: pd.DataFrame,
    addon_id: str,
    raw_rows: int,
    shutdown_removed: int,
    range_violations: List[Dict[str, Any]],
    spike_count: int,
    flatline_count: int,
    interpolated_count: int,
    cross_issues: List[Any]
) -> QualityReport:
    """Factory to build a QualityReport from pipeline state."""
    completeness = (
        1.0 - (df.isna().sum().sum() / (df.shape[0] * df.shape[1]))
        if df.shape[0] > 0 else 0.0
    )

    return QualityReport(
        addon_id=addon_id,
        raw_rows=raw_rows,
        final_rows=len(df),
        completeness=completeness,
        shutdown_removed=shutdown_removed,
        spike_count=spike_count,
        flatline_count=flatline_count,
        interpolated_count=interpolated_count,
        range_violations=range_violations,
        cross_issues=cross_issues
    )
