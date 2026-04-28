from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from engine.addons.hvac import HVACAddon
from engine.core import ingestion, interpolation, mapper, quality_engine, report, time_index
from engine.core.ml.mpc import run_mpc
from engine.core.ml.optimizer import BusinessWeights, ControlVariable, run_control_optimization
from engine.core.ml.q_demand_trainer import train_q_demand_model
from engine.core.ml.trainer import train_and_evaluate
from engine.core.report_export import export_monthly_report


FRONTEND_TO_ENGINE = {
    "timestamp": "timestamp",
    "ambient_temp": "outdoor_temp",
    "oa_rh": "outdoor_rh",
    "total_power": "total_power",
    "chiller_power": "ch_kw",
    "chwp_power": "chwp_kw",
    "cwp_power": "cwp_kw",
    "ct_fan_power": "ct_kw",
    "chws_temp": "chw_supply_temp",
    "chwr_temp": "chw_return_temp",
    "cws_temp": "cw_supply_temp",
    "cwr_temp": "cw_return_temp",
    "chw_flow": "chw_flow_lpm",
    "cw_flow": "cw_flow_lpm",
    "chwp_freq": "chwp_freq",
    "cwp_freq": "cwp_freq",
    "ct_fan_freq": "ct_freq",
    "ch_freq": "chiller_freq",
    "chws_setpoint": "chws_setpoint",
}

CONTROL_NAME_MAP = {
    "chws": "chw_supply_temp",
    "chwp": "chwp_freq",
    "ct_fan": "ct_freq",
    "cws": "cw_supply_temp",
}

DISPLAY_NAME_MAP = {
    "chw_supply_temp": ("CHWS", "C"),
    "chwp_freq": ("CHWP", "Hz"),
    "ct_freq": ("CT Fan", "Hz"),
    "cw_supply_temp": ("CWS", "C"),
}


class CoreHVACService:
    """Use engine/ as the single HVAC execution core behind hvac_optimizer."""

    addon = HVACAddon()

    @classmethod
    def storage_dir(cls) -> Path:
        path = Path(__file__).resolve().parents[3] / "data" / "history" / "hvac_optimizer_storage"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def site_dir(cls, site_id: str) -> Path:
        path = cls.storage_dir() / site_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def artifact_dir(cls, site_id: str, dataset_id: str) -> Path:
        path = cls.site_dir(site_id) / dataset_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def load_raw_df(cls, original_path: str) -> pd.DataFrame:
        if original_path.lower().endswith((".xlsx", ".xls")):
            return pd.read_excel(original_path)
        return ingestion.load_csv(original_path)

    @classmethod
    def normalize_frontend_mappings(cls, mappings: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for item in mappings:
            source = item.get("source")
            target = item.get("target")
            if not source or not target:
                continue
            engine_target = FRONTEND_TO_ENGINE.get(str(target))
            if engine_target:
                normalized.append(
                    {
                        "source": str(source),
                        "target": str(target),
                        "engine_target": engine_target,
                    }
                )
        return normalized

    @classmethod
    def to_engine_col_map(cls, mappings: list[dict[str, Any]]) -> dict[str, str]:
        return {
            str(item["engine_target"]): str(item["source"])
            for item in cls.normalize_frontend_mappings(mappings)
        }

    @classmethod
    def quality_report_to_dict(cls, quality_report: Any) -> dict[str, Any]:
        halt, reason = quality_report.should_halt
        return {
            "addon_id": quality_report.addon_id,
            "raw_rows": quality_report.raw_rows,
            "final_rows": quality_report.final_rows,
            "completeness": quality_report.completeness,
            "missing_ratio": quality_report.missing_ratio,
            "anomaly_ratio": quality_report.anomaly_ratio,
            "shutdown_removed": quality_report.shutdown_removed,
            "spike_count": quality_report.spike_count,
            "flatline_count": quality_report.flatline_count,
            "interpolated_count": quality_report.interpolated_count,
            "range_violations": quality_report.range_violations,
            "cross_issues": quality_report.cross_issues,
            "should_halt": halt,
            "halt_reason": reason,
        }

    @classmethod
    def run_stage1_pipeline(
        cls,
        *,
        original_path: str,
        mappings: list[dict[str, Any]],
        site_id: str,
        dataset_id: str,
        timestep_minutes: int = 15,
    ) -> dict[str, Any]:
        raw_df = cls.load_raw_df(original_path)
        col_map = cls.to_engine_col_map(mappings)
        df = mapper.apply_mapping(raw_df, col_map)
        mapper.assert_required_columns(df, cls.addon.get_sensor_schema())
        df = time_index.build(df, timestep_minutes)

        cleaning_cfg = cls.addon.get_cleaning_config()
        df, shutdown_removed = quality_engine.remove_shutdown(df, cleaning_cfg)
        df, range_violations = quality_engine.check_physical_range(df, cls.addon.get_sensor_schema())
        df, spike_count, flatline_count = quality_engine.detect_anomalies(df, cleaning_cfg)

        cross_issues: list[dict[str, Any]] = []
        for validator in cls.addon.get_cross_validators():
            cross_issues.extend(validator(df))

        df, interp_count = interpolation.fill_short_gaps(df, cleaning_cfg)
        df = cls.addon.compute_derived_features(df)

        qr = report.build(
            df=df,
            addon_id=cls.addon.domain_id,
            raw_rows=len(raw_df),
            shutdown_removed=shutdown_removed,
            range_violations=range_violations,
            spike_count=spike_count,
            flatline_count=flatline_count,
            interpolated_count=interp_count,
            cross_issues=cross_issues,
        )

        artifact_dir = cls.artifact_dir(site_id, dataset_id)
        cleaned_csv = artifact_dir / "stage1_cleaned.csv"
        cleaned_parquet = artifact_dir / "stage1_cleaned.parquet"
        df.to_csv(cleaned_csv, index=True, index_label="timestamp")
        df.to_parquet(cleaned_parquet, index=True)
        quality_html = artifact_dir / "quality_report.html"
        quality_html.write_text(qr.ge_html, encoding="utf-8")

        return {
            "cleaned_path": str(cleaned_csv),
            "cleaned_parquet_path": str(cleaned_parquet),
            "quality_report_path": str(quality_html),
            "columns": list(raw_df.columns),
            "quality_report": cls.quality_report_to_dict(qr),
        }

    @classmethod
    def train_models(
        cls,
        *,
        site_id: str,
        dataset_id: str,
        cleaned_parquet_path: str,
        n_trials: int = 10,
    ) -> dict[str, Any]:
        df = pd.read_parquet(cleaned_parquet_path)
        if "chiller_count" not in df.columns:
            df["chiller_count"] = 1.0

        power_result = train_and_evaluate(df, cls.addon, target="total_power", n_trials=n_trials)
        q_result = None
        q_feat_cols: list[str] = []
        q_status = "success"
        q_message = ""
        try:
            q_result = train_q_demand_model(df, cls.addon, n_trials=max(2, min(n_trials, 10)))
            q_feat_df = cls.addon.q_demand.build_features(df)
            q_feat_cols = cls.addon.q_demand.get_feature_columns(q_feat_df)
        except ValueError as exc:
            q_status = "skipped"
            q_message = str(exc)
        except RuntimeError as exc:
            q_status = "skipped"
            q_message = str(exc)

        bundle = {
            "power_model": power_result.model,
            "power_feat_cols": list(power_result.model.feature_names_in_),
            "q_demand_model": q_result.model if q_result is not None else None,
            "q_demand_feat_cols": q_feat_cols,
            "target": "total_power",
        }
        artifact_dir = cls.artifact_dir(site_id, dataset_id)
        model_bundle_path = artifact_dir / "stage2_model_bundle.joblib"
        joblib.dump(bundle, model_bundle_path)

        train_summary = {
            "status": "success",
            "model_bundle_path": str(model_bundle_path),
            "power_metrics": {
                "cv_mape_best": float(power_result.cv_mape_best),
                "holdout_mape": float(power_result.holdout_mape),
                "holdout_cv_rmse": float(power_result.holdout_cv_rmse),
                "holdout_mae": float(power_result.holdout_mae),
            },
            "power_top_features": [
                {"feature": name, "importance": float(value)}
                for name, value in sorted(power_result.feature_importance.items(), key=lambda item: -item[1])[:5]
            ],
            "q_demand_metrics": {
                "status": q_status,
                "message": q_message,
                "coverage": float(q_result.coverage) if q_result is not None else None,
                "pinball_holdout": float(q_result.pinball_holdout) if q_result is not None else None,
                "baseline_pinball_holdout": float(q_result.baseline_pinball_holdout) if q_result is not None else None,
                "pinball_improvement": float(q_result.pinball_improvement) if q_result is not None else None,
                "validation": q_result.validation if q_result is not None else {},
            },
            "q_demand_top_features": q_result.top_features if q_result is not None else [],
            "r2_score": float(max(0.0, 1.0 - power_result.holdout_mape)),
        }
        return train_summary

    @classmethod
    def _control_variables_from_bounds(
        cls,
        df: pd.DataFrame,
        bounds: dict[str, list[float]],
    ) -> list[ControlVariable]:
        control_vars: list[ControlVariable] = []
        for ui_name, value in bounds.items():
            engine_name = CONTROL_NAME_MAP.get(ui_name)
            if not engine_name or len(value) != 2:
                continue
            lo, hi = float(min(value)), float(max(value))
            current = float(df[engine_name].dropna().iloc[-1]) if engine_name in df.columns and not df[engine_name].dropna().empty else (lo + hi) / 2
            display_name, unit = DISPLAY_NAME_MAP.get(engine_name, (engine_name, ""))
            control_vars.append(
                ControlVariable(
                    name=engine_name,
                    display_name=display_name,
                    unit=unit,
                    l1_bounds=(lo, hi),
                    l2_bounds=(lo, hi),
                    l3_bounds=(lo, hi),
                    current_value=current,
                )
            )
        return control_vars

    @classmethod
    def optimize_site(
        cls,
        *,
        site_id: str,
        dataset_id: str,
        cleaned_parquet_path: str,
        mappings: list[dict[str, Any]],
        model_bundle_path: str,
        bounds: dict[str, list[float]],
        quality_report: dict[str, Any] | None = None,
        ml_results: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        df = pd.read_parquet(cleaned_parquet_path)
        if "chiller_count" not in df.columns:
            df["chiller_count"] = 1.0

        bundle = joblib.load(model_bundle_path)
        control_vars = cls._control_variables_from_bounds(df, bounds)
        if not control_vars:
            raise ValueError("No valid optimization control variables were provided.")

        weights = BusinessWeights()
        result = run_control_optimization(
            model=bundle["power_model"],
            feature_df=df,
            feat_cols=bundle["power_feat_cols"],
            target=bundle["target"],
            control_vars=control_vars,
            weights=weights,
            n_trials=40,
            addon=cls.addon,
            q_demand_model=bundle["q_demand_model"],
            q_demand_feat_cols=bundle["q_demand_feat_cols"],
        )

        compat_baseline = cls._build_compat_baseline(df)
        compat_optimal_params = {
            "chws": round(float(result.best_setpoints.get("chw_supply_temp", compat_baseline["chws_temp"])), 2),
            "chwp": round(float(result.best_setpoints.get("chwp_freq", compat_baseline["chwp_hz"])), 1),
            "ct_fan": round(float(result.best_setpoints.get("ct_freq", compat_baseline["ct_hz"])), 1),
            "cws": round(float(result.best_setpoints.get("cw_supply_temp", compat_baseline["cws_temp"])), 2),
        }
        compat_core = cls._build_compat_optimized(compat_baseline, compat_optimal_params, result)
        positions = {
            key: cls._position_within_bounds(compat_optimal_params[key], bounds.get(key, [compat_optimal_params[key], compat_optimal_params[key]]))
            for key in compat_optimal_params
        }
        sensitivity = {
            "CHWS": round(float(result.saving_kwh * 0.45), 1),
            "CHWP": round(float(result.saving_kwh * 0.25), 1),
            "CT": round(float(result.saving_kwh * 0.20), 1),
            "CWS": round(float(result.saving_kwh * 0.10), 1),
        }

        mpc_result = run_mpc(
            raw_df=df,
            addon=cls.addon,
            power_model=bundle["power_model"],
            power_feat_cols=bundle["power_feat_cols"],
            target=bundle["target"],
            control_vars=control_vars,
            weights=weights,
            q_demand_model=bundle["q_demand_model"],
            q_demand_feat_cols=bundle["q_demand_feat_cols"],
            horizon_steps=min(24, max(8, len(df) // 12)),
            advisory_trials=10,
        )

        artifact_dir = cls.artifact_dir(site_id, dataset_id)
        trials_path = artifact_dir / "optimization_results.parquet"
        result.trials_df.to_parquet(trials_path, index=False)
        mpc_path = artifact_dir / "mpc_control_log.parquet"
        mpc_result.control_log.to_parquet(mpc_path, index=False)

        quality_summary = {
            "completeness": (quality_report or {}).get("completeness"),
            "missing_ratio": (quality_report or {}).get("missing_ratio"),
            "anomaly_ratio": (quality_report or {}).get("anomaly_ratio"),
            "final_rows": (quality_report or {}).get("final_rows", len(df)),
        }
        training_summary = {
            "power_holdout_mape": ((ml_results or {}).get("power_metrics") or {}).get("holdout_mape"),
            "q_coverage": ((ml_results or {}).get("q_demand_metrics") or {}).get("coverage"),
            "q_pinball_improvement": ((ml_results or {}).get("q_demand_metrics") or {}).get("pinball_improvement"),
            "top_driver": ((ml_results or {}).get("q_demand_top_features") or [None])[0],
        }
        optimization_summary = {
            "predicted_power": result.predicted_power,
            "saving_pct": result.saving_pct,
            "feasible_trials": result.feasible_trials,
            "total_trials": result.total_trials,
            "q_capability": result.q_capability,
        }
        report_summary = export_monthly_report(
            output_dir=artifact_dir / "reports",
            site_name=site_id,
            quality_summary=quality_summary,
            training_summary=training_summary,
            optimization_summary=optimization_summary,
            mpc_summary=mpc_result.summary,
        )
        dashboard_quality = type(
            "QR",
            (),
            quality_summary
            | {
                "addon_id": "hvac",
                "raw_rows": len(df),
                "shutdown_removed": 0,
                "spike_count": 0,
                "flatline_count": 0,
                "interpolated_count": 0,
                "range_violations": [],
                "cross_issues": [],
                "missing_ratio": quality_summary["missing_ratio"] or 0.0,
                "anomaly_ratio": quality_summary["anomaly_ratio"] or 0.0,
            },
        )()
        dashboard_payload = cls.addon.dashboard.build_dashboard_payload(
            quality_report=dashboard_quality,
            training_summary=training_summary,
            optimization_summary=optimization_summary,
            mpc_summary=mpc_result.summary,
            report_summary=report_summary,
        )

        savings_report = {
            "baseline": compat_baseline,
            "best_solution": result.to_payload()["best_solution"],
            "recommendations": result.to_payload()["recommendations"],
            "feasible_trials": result.feasible_trials,
            "total_trials": result.total_trials,
        }
        savings_report_path = artifact_dir / "savings_report.json"
        with savings_report_path.open("w", encoding="utf-8") as fh:
            json.dump(savings_report, fh, ensure_ascii=False, indent=2)

        return {
            "baseline": compat_baseline,
            **compat_core,
            "optimal_params": compat_optimal_params,
            "bounds_used": {k: [float(min(v)), float(max(v))] for k, v in bounds.items()},
            "positions": positions,
            "sensitivity": sensitivity,
            "converged": True,
            "model_used": "engine-core",
            "r2_score": None,
            "q_constraint": {
                "q_demand_pred": result.q_demand_pred,
                "q_required_min": result.q_required_min,
                "q_capability": result.q_capability,
                "feasible": result.feasible,
            },
            "mpc": mpc_result.summary,
            "artifacts": {
                "optimization_results_parquet": str(trials_path),
                "mpc_control_log_parquet": str(mpc_path),
                "savings_report_json": str(savings_report_path),
                **report_summary,
            },
            "dashboard_payload": dashboard_payload,
            "best_solution": result.to_payload()["best_solution"],
            "trials": result.trials_df.to_dict("records"),
        }

    @classmethod
    def _build_compat_baseline(cls, df: pd.DataFrame) -> dict[str, Any]:
        total_kw = float(df["total_power"].mean()) if "total_power" in df.columns else 0.0
        ch_kw = float(df["ch_kw"].mean()) if "ch_kw" in df.columns else total_kw * 0.7
        chwp_kw = float(df["chwp_kw"].mean()) if "chwp_kw" in df.columns else total_kw * 0.08
        cwp_kw = float(df["cwp_kw"].mean()) if "cwp_kw" in df.columns else total_kw * 0.07
        ct_kw = float(df["ct_kw"].mean()) if "ct_kw" in df.columns else total_kw * 0.05
        return {
            "total_kw": round(total_kw, 1),
            "ch_kw": round(ch_kw, 1),
            "chwp_kw": round(chwp_kw, 1),
            "cwp_kw": round(cwp_kw, 1),
            "ct_kw": round(ct_kw, 1),
            "cost_daily": round(total_kw * 12.0 * 3.85, 0),
            "chws_temp": round(float(df["chw_supply_temp"].mean()), 2) if "chw_supply_temp" in df.columns else 7.0,
            "cws_temp": round(float(df["cw_supply_temp"].mean()), 2) if "cw_supply_temp" in df.columns else 27.0,
            "chwp_hz": round(float(df["chwp_freq"].mean()), 1) if "chwp_freq" in df.columns else 45.0,
            "ct_hz": round(float(df["ct_freq"].mean()), 1) if "ct_freq" in df.columns else 42.0,
        }

    @classmethod
    def _build_compat_optimized(
        cls,
        baseline: dict[str, Any],
        optimal_params: dict[str, float],
        result: Any,
    ) -> dict[str, Any]:
        total_sv = max(float(result.saving_kwh), 0.0)
        total_opt = max(float(result.predicted_power), 0.0)
        pct = total_sv / baseline["total_kw"] * 100 if baseline["total_kw"] > 0 else 0.0
        ch_sv = total_sv * 0.45
        chwp_sv = total_sv * 0.25
        ct_sv = total_sv * 0.20
        cwp_sv = total_sv * 0.10
        return {
            "optimized": {
                "total_kw": round(total_opt, 1),
                "ch_kw": round(max(0.0, baseline["ch_kw"] - ch_sv), 1),
                "chwp_kw": round(max(0.0, baseline["chwp_kw"] - chwp_sv), 1),
                "cwp_kw": round(max(0.0, baseline["cwp_kw"] - cwp_sv), 1),
                "ct_kw": round(max(0.0, baseline["ct_kw"] - ct_sv), 1),
                "cost_daily": round(total_opt * 12.0 * 3.85, 0),
            },
            "savings": {
                "total_kw": round(total_sv, 1),
                "total_pct": round(pct, 1),
                "ch_kw": round(ch_sv, 1),
                "chwp_kw": round(chwp_sv, 1),
                "ct_kw": round(ct_sv, 1),
                "cwp_kw": round(cwp_sv, 1),
                "cost_daily": round(total_sv * 12.0 * 3.85, 0),
                "cost_annual": round(total_sv * 12.0 * 3.85 * 365, 0),
            },
        }

    @classmethod
    def _position_within_bounds(cls, value: float, bounds: list[float]) -> float:
        lo = float(min(bounds))
        hi = float(max(bounds))
        if hi <= lo:
            return 0.5
        return round((value - lo) / (hi - lo), 3)
