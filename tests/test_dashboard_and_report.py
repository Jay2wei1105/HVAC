from __future__ import annotations

from pathlib import Path
import shutil

from engine.addons.hvac import HVACAddon
from engine.core.report import QualityReport
from engine.core.report_export import export_monthly_report, render_monthly_report_html


def _quality_report() -> QualityReport:
    return QualityReport(
        addon_id="hvac",
        raw_rows=100,
        final_rows=96,
        completeness=0.97,
        shutdown_removed=2,
        spike_count=1,
        flatline_count=1,
        interpolated_count=4,
    )


def test_hvac_dashboard_payload_has_five_sections() -> None:
    addon = HVACAddon()
    payload = addon.dashboard.build_dashboard_payload(
        quality_report=_quality_report(),
        training_summary={
            "power_holdout_mape": 0.12,
            "q_coverage": 0.89,
            "q_pinball_improvement": 0.34,
            "top_driver": "outdoor_temp",
        },
        optimization_summary={
            "predicted_power": 310.0,
            "saving_pct": 0.08,
            "feasible_trials": 17,
            "q_capability": 420.0,
        },
        mpc_summary={
            "total_energy_baseline": 1200.0,
            "total_energy_mpc": 1100.0,
            "total_energy_oracle": 1080.0,
            "saving_ratio_vs_oracle": 0.83,
        },
        report_summary={
            "html_path": "report.html",
            "pdf_path": None,
            "template_version": "v1",
        },
    )
    assert set(payload) == {"quality", "modeling", "optimization", "mpc", "report"}
    assert payload["quality"][0]["title"] == "Completeness"
    assert payload["modeling"][0]["title"] == "Power MAPE"


def test_export_monthly_report_writes_html() -> None:
    html = render_monthly_report_html(
        site_name="site_default",
        quality_summary={"completeness": 0.97, "missing_ratio": 0.03, "anomaly_ratio": 0.02, "final_rows": 96},
        training_summary={"power_holdout_mape": 0.12, "q_coverage": 0.89, "q_pinball_improvement": 0.34, "top_driver": "outdoor_temp"},
        optimization_summary={"predicted_power": 310.0, "saving_pct": 0.08, "feasible_trials": 17, "total_trials": 20, "q_capability": 420.0},
        mpc_summary={"total_energy_baseline": 1200.0, "total_energy_mpc": 1100.0, "total_energy_oracle": 1080.0, "saving_ratio_vs_oracle": 0.83, "passed_ratio_gate": False},
    )
    assert "site_default EMS Report" in html
    assert "MPC Replay" in html


def test_export_monthly_report_creates_artifacts() -> None:
    tmp_path = Path("data/history/_test_report_output")
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    result = export_monthly_report(
        output_dir=tmp_path,
        site_name="site_default",
        quality_summary={"completeness": 0.97, "missing_ratio": 0.03, "anomaly_ratio": 0.02, "final_rows": 96},
        training_summary={"power_holdout_mape": 0.12, "q_coverage": 0.89, "q_pinball_improvement": 0.34, "top_driver": "outdoor_temp"},
        optimization_summary={"predicted_power": 310.0, "saving_pct": 0.08, "feasible_trials": 17, "total_trials": 20, "q_capability": 420.0},
    )
    assert Path(result["html_path"]).exists()
    assert result["template_version"] == "v1"
