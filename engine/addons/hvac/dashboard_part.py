from __future__ import annotations

from engine.core.dashboard import DashboardPayload, WidgetSpec


class HVACDashboardPart:
    """Minimal HVAC dashboard payload builder for Stage 5."""

    def build_dashboard_payload(
        self,
        *,
        quality_report: object | None = None,
        training_summary: dict | None = None,
        optimization_summary: dict | None = None,
        mpc_summary: dict | None = None,
        report_summary: dict | None = None,
    ) -> dict:
        payload = DashboardPayload()

        if quality_report is not None:
            payload.quality.extend(
                [
                    WidgetSpec("quality", "Completeness", "metric", round(quality_report.completeness, 4), unit="ratio"),
                    WidgetSpec("quality", "Missing Ratio", "metric", round(quality_report.missing_ratio, 4), unit="ratio"),
                    WidgetSpec("quality", "Anomaly Ratio", "metric", round(quality_report.anomaly_ratio, 4), unit="ratio"),
                    WidgetSpec("quality", "Rows", "metric", quality_report.final_rows, unit="rows"),
                ]
            )

        if training_summary:
            payload.modeling.extend(
                [
                    WidgetSpec("modeling", "Power MAPE", "metric", training_summary.get("power_holdout_mape")),
                    WidgetSpec("modeling", "Q Coverage", "metric", training_summary.get("q_coverage")),
                    WidgetSpec("modeling", "Q Pinball Improvement", "metric", training_summary.get("q_pinball_improvement")),
                    WidgetSpec("modeling", "Top Driver", "text", training_summary.get("top_driver")),
                ]
            )

        if optimization_summary:
            payload.optimization.extend(
                [
                    WidgetSpec("optimization", "Predicted Power", "metric", optimization_summary.get("predicted_power"), unit="kW"),
                    WidgetSpec("optimization", "Saving Percent", "metric", optimization_summary.get("saving_pct"), unit="ratio"),
                    WidgetSpec("optimization", "Feasible Trials", "metric", optimization_summary.get("feasible_trials")),
                    WidgetSpec("optimization", "Q Capability", "metric", optimization_summary.get("q_capability"), unit="kW"),
                ]
            )

        if mpc_summary:
            payload.mpc.extend(
                [
                    WidgetSpec("mpc", "Baseline Energy", "metric", mpc_summary.get("total_energy_baseline"), unit="kWh"),
                    WidgetSpec("mpc", "MPC Energy", "metric", mpc_summary.get("total_energy_mpc"), unit="kWh"),
                    WidgetSpec("mpc", "Oracle Energy", "metric", mpc_summary.get("total_energy_oracle"), unit="kWh"),
                    WidgetSpec("mpc", "Savings Ratio vs Oracle", "metric", mpc_summary.get("saving_ratio_vs_oracle"), unit="ratio"),
                ]
            )

        if report_summary:
            payload.report.extend(
                [
                    WidgetSpec("report", "HTML Report", "text", report_summary.get("html_path")),
                    WidgetSpec("report", "PDF Report", "text", report_summary.get("pdf_path")),
                    WidgetSpec("report", "Version", "text", report_summary.get("template_version", "v1")),
                ]
            )

        return payload.to_dict()

