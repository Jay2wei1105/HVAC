from __future__ import annotations

from pathlib import Path
from typing import Any


def render_monthly_report_html(
    *,
    site_name: str,
    quality_summary: dict[str, Any],
    training_summary: dict[str, Any],
    optimization_summary: dict[str, Any],
    mpc_summary: dict[str, Any] | None = None,
) -> str:
    """Render a lightweight Stage 5 monthly report as HTML."""
    mpc_block = ""
    if mpc_summary:
        mpc_block = f"""
        <h2>MPC Replay</h2>
        <ul>
          <li>Baseline energy: {mpc_summary.get("total_energy_baseline")}</li>
          <li>MPC energy: {mpc_summary.get("total_energy_mpc")}</li>
          <li>Oracle energy: {mpc_summary.get("total_energy_oracle")}</li>
          <li>Savings ratio vs oracle: {mpc_summary.get("saving_ratio_vs_oracle")}</li>
          <li>Passed ratio gate: {mpc_summary.get("passed_ratio_gate")}</li>
        </ul>
        """

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{site_name} EMS Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #1f2937; }}
    h1, h2 {{ margin-bottom: 8px; }}
    section {{ margin-bottom: 24px; }}
    ul {{ margin: 8px 0 0 20px; }}
    .meta {{ color: #4b5563; }}
  </style>
</head>
<body>
  <h1>{site_name} EMS Report</h1>
  <p class="meta">Template version: v1</p>
  <section>
    <h2>Data Quality</h2>
    <ul>
      <li>Completeness: {quality_summary.get("completeness")}</li>
      <li>Missing ratio: {quality_summary.get("missing_ratio")}</li>
      <li>Anomaly ratio: {quality_summary.get("anomaly_ratio")}</li>
      <li>Rows: {quality_summary.get("final_rows")}</li>
    </ul>
  </section>
  <section>
    <h2>Modeling</h2>
    <ul>
      <li>Power holdout MAPE: {training_summary.get("power_holdout_mape")}</li>
      <li>Q-demand coverage: {training_summary.get("q_coverage")}</li>
      <li>Q-demand pinball improvement: {training_summary.get("q_pinball_improvement")}</li>
      <li>Top driver: {training_summary.get("top_driver")}</li>
    </ul>
  </section>
  <section>
    <h2>Optimization</h2>
    <ul>
      <li>Predicted power: {optimization_summary.get("predicted_power")}</li>
      <li>Saving percent: {optimization_summary.get("saving_pct")}</li>
      <li>Feasible trials: {optimization_summary.get("feasible_trials")} / {optimization_summary.get("total_trials")}</li>
      <li>Q capability: {optimization_summary.get("q_capability")}</li>
    </ul>
  </section>
  {mpc_block}
</body>
</html>"""


def export_monthly_report(
    *,
    output_dir: str | Path,
    site_name: str,
    quality_summary: dict[str, Any],
    training_summary: dict[str, Any],
    optimization_summary: dict[str, Any],
    mpc_summary: dict[str, Any] | None = None,
) -> dict[str, str | None]:
    """Write HTML and, when available, PDF report artifacts."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    html = render_monthly_report_html(
        site_name=site_name,
        quality_summary=quality_summary,
        training_summary=training_summary,
        optimization_summary=optimization_summary,
        mpc_summary=mpc_summary,
    )
    html_path = output_path / "monthly_report.html"
    html_path.write_text(html, encoding="utf-8")

    pdf_path: Path | None = None
    try:
        from weasyprint import HTML  # type: ignore

        pdf_path = output_path / "monthly_report.pdf"
        HTML(string=html).write_pdf(str(pdf_path))
    except Exception:
        pdf_path = None

    return {
        "html_path": str(html_path),
        "pdf_path": str(pdf_path) if pdf_path else None,
        "template_version": "v1",
    }

