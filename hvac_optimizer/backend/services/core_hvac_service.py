from __future__ import annotations

import json
import os
from itertools import product
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import joblib
import pandas as pd

from engine.addons.hvac import HVACAddon
from engine.core import contracts, ingestion, interpolation, mapper, quality_engine, report, time_index
from engine.core.ml.mpc import run_mpc
from engine.core.ml.optimizer import (
    BusinessWeights,
    ControlVariable,
    _append_trial_row,
    _ensure_datetime_index,
    _infer_step,
    _predict_from_last_row,
    _prepare_prediction_frame,
    _prepare_q_demand_frame,
    run_control_optimization,
)
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
    "cwp": "cwp_freq",
    "ct_fan": "ct_freq",
}

DISPLAY_NAME_MAP = {
    "chw_supply_temp": ("CHWS", "C"),
    "chwp_freq": ("CHWP", "Hz"),
    "cwp_freq": ("CWP", "Hz"),
    "ct_freq": ("CT Fan", "Hz"),
}

OPT_INTERVAL_HISTORY_ROWS = 104
LEGACY_OPT_TRIALS = 16
FAST_MODE_LEGACY_OPT_TRIALS = 8
MPC_ADVISORY_TRIALS = 6
MPC_HORIZON_MAX = 12
PROJECTION_ACTION_LIMIT = 48
FAST_MODE_TOP_ACTIONS = 36
FAST_MODE_REPRESENTATIVE_POINTS = 4
EXTREME_MODE_TOP_ACTIONS = 16
EXTREME_MODE_REPRESENTATIVE_POINTS = 3


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
            "contract_issues": quality_report.contract_issues,
            "cleaning_actions": quality_report.cleaning_actions,
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
        preclean = contracts.preclean_raw_frame(raw_df)
        raw_df = preclean.df
        col_map = cls.to_engine_col_map(mappings)
        df = mapper.apply_mapping(raw_df, col_map)
        mapper.assert_required_columns(df, cls.addon.get_sensor_schema())
        contract_result = contracts.coerce_and_validate_sensor_frame(df, cls.addon.get_sensor_schema())
        df = contract_result.df
        df = time_index.build(df, timestep_minutes)
        time_contract_issues = contracts.validate_time_index(df)

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
            contract_issues=contract_result.issues + time_contract_issues,
            cleaning_actions=preclean.stats | contract_result.stats,
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
            "q_demand_feature_importance": [
                {"feature": name, "importance": float(value)}
                for name, value in sorted(
                    (q_result.feature_importance if q_result is not None else {}).items(),
                    key=lambda item: -item[1],
                )[:8]
            ],
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
        optimization_mode: str = "standard",
        quality_report: dict[str, Any] | None = None,
        ml_results: dict[str, Any] | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        def report_progress(payload: dict[str, Any] | None = None, **kwargs: Any) -> None:
            if progress_callback is not None:
                merged = dict(payload or {})
                merged.update(kwargs)
                progress_callback(merged)

        report_progress(
            percent=2,
            stage="loading",
            message="正在載入清洗後資料與訓練模型...",
        )
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
            n_trials=FAST_MODE_LEGACY_OPT_TRIALS if optimization_mode == "fast" else LEGACY_OPT_TRIALS,
            addon=cls.addon,
            q_demand_model=bundle["q_demand_model"],
            q_demand_feat_cols=bundle["q_demand_feat_cols"],
        )
        report_progress(
            percent=12,
            stage="grid_build",
            message="正在建立粗網格控制組合...",
        )
        candidate_actions = cls._build_grid_actions(control_vars=control_vars)
        if not candidate_actions:
            raise ValueError("No grid control combinations could be built from the provided bounds.")
        interval_actions = candidate_actions
        if optimization_mode == "fast":
            report_progress(
                percent=14,
                stage="fast_prescreen",
                message="正在做加速版全域快篩，縮小候選控制組合...",
                control_combinations=int(len(candidate_actions)),
            )
            interval_actions = cls._prefilter_actions_for_interval(
                raw_df=df,
                addon=cls.addon,
                power_model=bundle["power_model"],
                power_feat_cols=bundle["power_feat_cols"],
                target=bundle["target"],
                control_vars=control_vars,
                candidate_actions=candidate_actions,
                q_demand_model=bundle["q_demand_model"],
                q_demand_feat_cols=bundle["q_demand_feat_cols"],
                top_actions=FAST_MODE_TOP_ACTIONS,
                representative_points=FAST_MODE_REPRESENTATIVE_POINTS,
            )
        elif optimization_mode == "extreme":
            report_progress(
                percent=14,
                stage="extreme_prescreen",
                message="正在做極速版全域快篩，縮到最少候選控制組合...",
                control_combinations=int(len(candidate_actions)),
            )
            interval_actions = cls._prefilter_actions_for_interval(
                raw_df=df,
                addon=cls.addon,
                power_model=bundle["power_model"],
                power_feat_cols=bundle["power_feat_cols"],
                target=bundle["target"],
                control_vars=control_vars,
                candidate_actions=candidate_actions,
                q_demand_model=bundle["q_demand_model"],
                q_demand_feat_cols=bundle["q_demand_feat_cols"],
                top_actions=EXTREME_MODE_TOP_ACTIONS,
                representative_points=EXTREME_MODE_REPRESENTATIVE_POINTS,
            )
        report_progress(
            percent=16,
            stage="interval_prepare",
            message="正在準備每小時取樣點與整段區間分析...",
            control_combinations=int(len(interval_actions)),
            full_control_combinations=int(len(candidate_actions)),
            optimization_mode=optimization_mode,
        )
        interval_analysis = cls._evaluate_interval_actions(
            raw_df=df,
            addon=cls.addon,
            power_model=bundle["power_model"],
            power_feat_cols=bundle["power_feat_cols"],
            target=bundle["target"],
            control_vars=control_vars,
            candidate_actions=interval_actions,
            q_demand_model=bundle["q_demand_model"],
            q_demand_feat_cols=bundle["q_demand_feat_cols"],
            electricity_rate=4.0,
            progress_callback=report_progress,
        )
        if optimization_mode == "extreme":
            future_projection = {"summary": {}, "curve": []}
        else:
            report_progress(
                percent=84,
                stage="future_projection",
                message="正在整理未來一倍時長的預測趨勢...",
            )
            future_projection = cls._project_future_mpc_like(
                raw_df=df,
                addon=cls.addon,
                power_model=bundle["power_model"],
                power_feat_cols=bundle["power_feat_cols"],
                target=bundle["target"],
                control_vars=control_vars,
                candidate_actions=interval_analysis.get("projection_actions") or candidate_actions[:PROJECTION_ACTION_LIMIT],
                q_demand_model=bundle["q_demand_model"],
                q_demand_feat_cols=bundle["q_demand_feat_cols"],
                progress_callback=report_progress,
            )

        compat_baseline = cls._build_compat_baseline(df)
        compat_optimal_params = {
            "chws": round(float(result.best_setpoints.get("chw_supply_temp", compat_baseline["chws_temp"])), 2),
            "chwp": round(float(result.best_setpoints.get("chwp_freq", compat_baseline["chwp_hz"])), 1),
            "cwp": round(float(result.best_setpoints.get("cwp_freq", compat_baseline["cwp_hz"])), 1),
            "ct_fan": round(float(result.best_setpoints.get("ct_freq", compat_baseline["ct_hz"])), 1),
        }
        compat_core = cls._build_compat_optimized(compat_baseline, compat_optimal_params, result)
        positions = {
            key: cls._position_within_bounds(compat_optimal_params[key], bounds.get(key, [compat_optimal_params[key], compat_optimal_params[key]]))
            for key in compat_optimal_params
        }
        sensitivity = {
            "CHWS": round(float(result.saving_kwh * 0.45), 1),
            "CHWP": round(float(result.saving_kwh * 0.25), 1),
            "CWP": round(float(result.saving_kwh * 0.10), 1),
            "CT": round(float(result.saving_kwh * 0.20), 1),
        }

        if optimization_mode == "extreme":
            mpc_log_df = pd.DataFrame()
            mpc_summary = {}
        else:
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
                horizon_steps=min(MPC_HORIZON_MAX, max(6, len(df) // 24)),
                advisory_trials=MPC_ADVISORY_TRIALS,
            )
            mpc_log_df = mpc_result.control_log
            mpc_summary = mpc_result.summary

        artifact_dir = cls.artifact_dir(site_id, dataset_id)
        report_progress(
            percent=92,
            stage="artifacts",
            message="正在寫入最佳化結果與報表產物...",
        )
        trials_path = artifact_dir / "optimization_results.parquet"
        result.trials_df.to_parquet(trials_path, index=False)
        mpc_path = artifact_dir / "mpc_control_log.parquet"
        mpc_log_df.to_parquet(mpc_path, index=False)

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
            mpc_summary=mpc_summary,
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
            mpc_summary=mpc_summary,
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

        report_progress(
            percent=100,
            stage="completed",
            message="區間優化完成。",
            rows_evaluated=interval_analysis["summary"].get("rows_evaluated"),
            interval_hours=interval_analysis["summary"].get("interval_hours"),
        )
        return {
            "baseline": compat_baseline,
            **compat_core,
            "optimal_params": compat_optimal_params,
            "bounds_used": {k: [float(min(v)), float(max(v))] for k, v in bounds.items()},
            "optimization_mode": optimization_mode,
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
            "mpc": mpc_summary,
            "artifacts": {
                "optimization_results_parquet": str(trials_path),
                "mpc_control_log_parquet": str(mpc_path),
                "savings_report_json": str(savings_report_path),
                **report_summary,
            },
            "dashboard_payload": dashboard_payload,
            "best_solution": result.to_payload()["best_solution"],
            "trials": result.trials_df.to_dict("records"),
            "interval_summary": interval_analysis["summary"],
            "interval_curve": interval_analysis["curve"],
            "equipment_savings_kwh": interval_analysis["equipment_savings_kwh"],
            "future_projection": future_projection,
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
            "chwp_hz": round(float(df["chwp_freq"].mean()), 1) if "chwp_freq" in df.columns else 45.0,
            "cwp_hz": round(float(df["cwp_freq"].mean()), 1) if "cwp_freq" in df.columns else 45.0,
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
    def _sample_hourly_points(cls, df: pd.DataFrame) -> pd.DataFrame:
        working = _ensure_datetime_index(df)
        if working.empty:
            return working
        # Use the last available row in each hour as the representative operating point.
        sampled = working.groupby(working.index.floor("1h"), sort=True).tail(1)
        return sampled.sort_index()

    @classmethod
    def _build_grid_actions(
        cls,
        *,
        control_vars: list[ControlVariable],
    ) -> list[dict[str, float]]:
        step_map = {
            "chw_supply_temp": 1.0,
            "chwp_freq": 2.5,
            "cwp_freq": 2.5,
            "ct_freq": 2.5,
        }
        value_lists: list[list[float]] = []
        ordered_vars: list[ControlVariable] = []

        for cv in control_vars:
            lo, hi = cv.effective_bounds
            step = step_map.get(cv.name)
            if step is None or hi < lo:
                continue
            values: list[float] = []
            current = float(lo)
            while current <= hi + 1e-9:
                values.append(round(current, 4))
                current += step
            if not values or values[-1] < hi - 1e-9:
                values.append(round(float(hi), 4))
            value_lists.append(values)
            ordered_vars.append(cv)

        if not ordered_vars:
            return []

        actions: list[dict[str, float]] = []
        for combo in product(*value_lists):
            actions.append({cv.name: float(value) for cv, value in zip(ordered_vars, combo, strict=True)})
        return actions

    @classmethod
    def _prefilter_actions_for_interval(
        cls,
        *,
        raw_df: pd.DataFrame,
        addon: Any,
        power_model: Any,
        power_feat_cols: list[str],
        target: str,
        control_vars: list[ControlVariable],
        candidate_actions: list[dict[str, float]],
        q_demand_model: Any | None,
        q_demand_feat_cols: list[str] | None,
        top_actions: int,
        representative_points: int,
    ) -> list[dict[str, float]]:
        working_df = _ensure_datetime_index(raw_df)
        sampled_df = cls._sample_hourly_points(working_df)
        if sampled_df.empty or len(candidate_actions) <= top_actions:
            return candidate_actions

        total_points = len(sampled_df)
        selected_indexes = sorted({int(round(i * (total_points - 1) / max(representative_points - 1, 1))) for i in range(min(representative_points, total_points))})
        selected_rows = sampled_df.iloc[selected_indexes]
        action_stats: dict[tuple[tuple[str, float], ...], dict[str, Any]] = {}

        for ts, _current in selected_rows.iterrows():
            history_df = working_df.loc[:ts]
            if history_df.empty:
                continue
            for action in candidate_actions:
                try:
                    score = cls._score_action(
                        history_df=history_df,
                        action=action,
                        addon=addon,
                        power_model=power_model,
                        power_feat_cols=power_feat_cols,
                        target=target,
                        control_vars=control_vars,
                        q_demand_model=q_demand_model,
                        q_demand_feat_cols=q_demand_feat_cols,
                    )
                except Exception:
                    continue
                action_key = tuple(sorted((str(k), round(float(v), 4)) for k, v in action.items()))
                stat = action_stats.setdefault(
                    action_key,
                    {"action": {k: float(v) for k, v in action.items()}, "count": 0, "feasible_count": 0, "power_sum": 0.0},
                )
                stat["count"] += 1
                stat["feasible_count"] += int(bool(score["feasible"]))
                stat["power_sum"] += float(score["predicted_power"])

        ranked_actions = sorted(
            action_stats.values(),
            key=lambda item: (-item["feasible_count"], item["power_sum"] / max(item["count"], 1)),
        )
        filtered = [item["action"] for item in ranked_actions[:top_actions]]
        return filtered or candidate_actions[:top_actions]

    @classmethod
    def _candidate_actions_from_trials(
        cls,
        *,
        control_vars: list[ControlVariable],
        trials_df: pd.DataFrame,
        max_actions: int = 20,
    ) -> list[dict[str, float]]:
        actions: list[dict[str, float]] = []
        seen: set[tuple[float, ...]] = set()

        def _add(action: dict[str, float]) -> None:
            key = tuple(round(float(action[cv.name]), 4) for cv in control_vars)
            if key in seen:
                return
            seen.add(key)
            actions.append({cv.name: float(action[cv.name]) for cv in control_vars})

        baseline_action = {cv.name: float(cv.current_value) for cv in control_vars}
        _add(baseline_action)

        if not trials_df.empty:
            ordered = trials_df.sort_values(by=["feasible", "objective"], ascending=[False, True], kind="stable")
            for _, row in ordered.iterrows():
                action = {}
                valid = True
                for cv in control_vars:
                    value = row.get(cv.name)
                    if pd.isna(value):
                        valid = False
                        break
                    action[cv.name] = float(value)
                if valid:
                    _add(action)
                if len(actions) >= max_actions:
                    break

        for ratio in (0.2, 0.5, 0.8):
            action = {}
            for cv in control_vars:
                lo, hi = cv.effective_bounds
                action[cv.name] = float(lo + (hi - lo) * ratio)
            _add(action)
            if len(actions) >= max_actions:
                break

        return actions[:max_actions]

    @classmethod
    def _score_action(
        cls,
        *,
        history_df: pd.DataFrame,
        action: dict[str, float],
        addon: Any,
        power_model: Any,
        power_feat_cols: list[str],
        target: str,
        control_vars: list[ControlVariable],
        q_demand_model: Any | None,
        q_demand_feat_cols: list[str] | None,
        history_rows: int = OPT_INTERVAL_HISTORY_ROWS,
    ) -> dict[str, Any]:
        scenario_raw_df = _append_trial_row(history_df, control_vars, action, history_rows=history_rows)
        predicted_power = _predict_from_last_row(
            power_model,
            _prepare_prediction_frame(scenario_raw_df, addon, target),
            power_feat_cols,
        )

        q_required_min = None
        q_capability = None
        q_gap_pct = None
        feasible = True
        q_demand_pred = None

        if q_demand_model is not None and q_demand_feat_cols:
            q_frame = _prepare_q_demand_frame(scenario_raw_df, addon)
            if q_frame.empty:
                feasible = False
            else:
                missing_q_cols = [col for col in q_demand_feat_cols if col not in q_frame.columns]
                if missing_q_cols:
                    feasible = False
                else:
                    q_row = q_frame.iloc[[-1]][q_demand_feat_cols]
                    q_demand_pred = float(q_demand_model.predict(q_row)[0])
                    q_required_min = q_demand_pred * 1.05
                    q_capability = addon.decision.estimate_q_capability(scenario_raw_df, action)
                    if q_capability is None or q_required_min is None or q_required_min <= 0:
                        feasible = False
                    else:
                        q_gap_pct = (float(q_capability) - float(q_required_min)) / float(q_required_min) * 100.0
                        feasible = bool(q_capability >= q_required_min)

        return {
            "predicted_power": float(predicted_power),
            "q_demand_pred": q_demand_pred,
            "q_required_min": q_required_min,
            "q_capability": q_capability,
            "q_gap_pct": q_gap_pct,
            "feasible": feasible,
            "action": action,
        }

    @classmethod
    def _evaluate_interval_actions(
        cls,
        *,
        raw_df: pd.DataFrame,
        addon: Any,
        power_model: Any,
        power_feat_cols: list[str],
        target: str,
        control_vars: list[ControlVariable],
        candidate_actions: list[dict[str, float]],
        q_demand_model: Any | None,
        q_demand_feat_cols: list[str] | None,
        electricity_rate: float,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        def report_progress(**payload: Any) -> None:
            if progress_callback is not None:
                progress_callback(payload)

        working_df = _ensure_datetime_index(raw_df)
        sampled_df = cls._sample_hourly_points(working_df)
        interval_h = max(_infer_step(working_df.index).total_seconds() / 3600.0, 0.0) if len(working_df.index) > 1 else 0.25
        equipment_cols = ["ch_kw", "chwp_kw", "cwp_kw", "ct_kw"]
        equipment_savings = {col: 0.0 for col in equipment_cols}
        rows: list[dict[str, Any]] = []
        total_points = int(len(sampled_df))
        action_stats: dict[tuple[tuple[str, float], ...], dict[str, Any]] = {}
        no_feasible_points = 0

        report_progress(
            percent=18,
            stage="interval_running",
            message="正在逐點評估控制組合...",
            completed_points=0,
            total_points=total_points,
        )

        for point_idx, (ts, current) in enumerate(sampled_df.iterrows(), start=1):
            history_df = working_df.loc[:ts]
            if history_df.empty:
                continue
            baseline_power = float(current.get(target, 0.0) or 0.0)

            best_score: dict[str, Any] | None = None
            fallback_score: dict[str, Any] | None = None
            baseline_action = {cv.name: float(current.get(cv.name, cv.current_value) or cv.current_value) for cv in control_vars}
            baseline_score: dict[str, Any] | None = None
            for action in candidate_actions:
                try:
                    score = cls._score_action(
                        history_df=history_df,
                        action=action,
                        addon=addon,
                        power_model=power_model,
                        power_feat_cols=power_feat_cols,
                        target=target,
                        control_vars=control_vars,
                        q_demand_model=q_demand_model,
                        q_demand_feat_cols=q_demand_feat_cols,
                    )
                except Exception:
                    continue

                if fallback_score is None or score["predicted_power"] < fallback_score["predicted_power"]:
                    fallback_score = score
                if score["feasible"] and (best_score is None or score["predicted_power"] < best_score["predicted_power"]):
                    best_score = score
                if action == baseline_action:
                    baseline_score = score

            if baseline_score is None:
                try:
                    baseline_score = cls._score_action(
                        history_df=history_df,
                        action=baseline_action,
                        addon=addon,
                        power_model=power_model,
                        power_feat_cols=power_feat_cols,
                        target=target,
                        control_vars=control_vars,
                        q_demand_model=q_demand_model,
                        q_demand_feat_cols=q_demand_feat_cols,
                    )
                except Exception:
                    baseline_score = {
                        "predicted_power": baseline_power,
                        "q_demand_pred": None,
                        "q_required_min": None,
                        "q_capability": None,
                        "q_gap_pct": None,
                        "feasible": False,
                        "action": baseline_action,
                    }

            if best_score is not None:
                chosen = best_score
                solution_type = "optimized"
            elif q_demand_model is not None and q_demand_feat_cols:
                chosen = baseline_score
                solution_type = "baseline_locked"
                no_feasible_points += 1
            else:
                chosen = fallback_score or baseline_score
                solution_type = "fallback"

            optimized_power = max(float(chosen["predicted_power"]), 0.0)
            if solution_type == "baseline_locked":
                optimized_power = baseline_power
            saving_power = max(baseline_power - optimized_power, 0.0)

            component_total = 0.0
            component_values: dict[str, float] = {}
            for col in equipment_cols:
                value = max(float(current.get(col, 0.0) or 0.0), 0.0)
                component_values[col] = value
                component_total += value
            if component_total > 0 and saving_power > 0:
                for col in equipment_cols:
                    equipment_savings[col] += saving_power * (component_values[col] / component_total) * interval_h

            rows.append(
                {
                    "timestamp": ts,
                    "baseline_power_kw": baseline_power,
                    "optimized_power_kw": optimized_power,
                    "saving_power_kw": saving_power,
                    "q_required_min": chosen["q_required_min"],
                    "q_capability": chosen["q_capability"],
                    "q_gap_pct": chosen["q_gap_pct"],
                    "feasible": bool(chosen["feasible"]),
                    "solution_type": solution_type,
                }
            )
            action_key = tuple(sorted((str(k), round(float(v), 4)) for k, v in chosen["action"].items()))
            stat = action_stats.setdefault(
                action_key,
                {
                    "action": {k: float(v) for k, v in chosen["action"].items()},
                    "count": 0,
                    "power_sum": 0.0,
                    "feasible_count": 0,
                },
            )
            stat["count"] += 1
            stat["power_sum"] += optimized_power
            stat["feasible_count"] += int(bool(chosen["feasible"]))
            if total_points > 0:
                percent = 18 + int((point_idx / total_points) * 62)
                report_progress(
                    percent=min(percent, 80),
                    stage="interval_running",
                    message="正在逐點評估控制組合...",
                    completed_points=point_idx,
                    total_points=total_points,
                    current_timestamp=pd.Timestamp(ts).isoformat() if pd.notna(ts) else None,
                )

        curve_df = pd.DataFrame(rows)
        sampled_interval_h = 1.0 if len(sampled_df) > 1 else interval_h
        baseline_kwh = float(curve_df["baseline_power_kw"].sum() * sampled_interval_h) if not curve_df.empty else 0.0
        optimized_kwh = float(curve_df["optimized_power_kw"].sum() * sampled_interval_h) if not curve_df.empty else 0.0
        saving_kwh = max(baseline_kwh - optimized_kwh, 0.0)
        saving_pct = saving_kwh / baseline_kwh * 100.0 if baseline_kwh > 0 else 0.0
        avg_q_gap_pct = float(curve_df["q_gap_pct"].dropna().mean()) if "q_gap_pct" in curve_df and curve_df["q_gap_pct"].notna().any() else 0.0
        feasible_ratio = float(curve_df["feasible"].mean()) if not curve_df.empty else 0.0
        ranked_actions = sorted(
            action_stats.values(),
            key=lambda item: (-item["count"], -(item["feasible_count"]), item["power_sum"] / max(item["count"], 1)),
        )
        projection_actions = [item["action"] for item in ranked_actions[:PROJECTION_ACTION_LIMIT]]
        if not projection_actions:
            projection_actions = candidate_actions[:PROJECTION_ACTION_LIMIT]

        return {
            "summary": {
                "baseline_kwh": round(baseline_kwh, 2),
                "optimized_kwh": round(optimized_kwh, 2),
                "saving_kwh": round(saving_kwh, 2),
                "saving_pct": round(saving_pct, 2),
                "saving_cost_ntd": round(saving_kwh * electricity_rate, 0),
                "avg_q_gap_pct": round(avg_q_gap_pct, 2),
                "feasible_ratio": round(feasible_ratio, 4),
                "no_feasible_points": int(no_feasible_points),
                "rows_evaluated": int(len(curve_df)),
                "interval_hours": round(sampled_interval_h * len(curve_df), 2),
                "control_combinations": int(len(candidate_actions)),
            },
            "curve": [
                {
                    **row,
                    "timestamp": pd.Timestamp(row["timestamp"]).isoformat() if pd.notna(row["timestamp"]) else None,
                }
                for row in curve_df.to_dict("records")
            ],
            "equipment_savings_kwh": {col: round(val, 2) for col, val in equipment_savings.items()},
            "projection_actions": projection_actions,
        }

    @classmethod
    def _project_future_mpc_like(
        cls,
        *,
        raw_df: pd.DataFrame,
        addon: Any,
        power_model: Any,
        power_feat_cols: list[str],
        target: str,
        control_vars: list[ControlVariable],
        candidate_actions: list[dict[str, float]],
        q_demand_model: Any | None,
        q_demand_feat_cols: list[str] | None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        def report_progress(**payload: Any) -> None:
            if progress_callback is not None:
                progress_callback(payload)

        working_df = _ensure_datetime_index(raw_df)
        if working_df.empty:
            return {"summary": {}, "curve": []}

        sampled_df = cls._sample_hourly_points(working_df)
        if sampled_df.empty:
            return {"summary": {}, "curve": []}

        future_template = sampled_df.copy()
        step = pd.Timedelta(hours=1)
        projected_history = working_df.copy()
        rows: list[dict[str, Any]] = []
        interval_h = 1.0
        total_steps = int(len(future_template))

        for idx in range(len(future_template)):
            template_row = future_template.iloc[idx].copy()
            next_ts = projected_history.index[-1] + step

            best_score: dict[str, Any] | None = None
            fallback_score: dict[str, Any] | None = None
            for action in candidate_actions:
                scenario_raw_df = projected_history.iloc[-192:].copy()
                next_row = template_row.copy()
                for cv in control_vars:
                    next_row[cv.name] = float(action.get(cv.name, next_row.get(cv.name, cv.current_value)))
                scenario_raw_df = pd.concat([scenario_raw_df, pd.DataFrame([next_row], index=[next_ts])])
                try:
                    predicted_power = _predict_from_last_row(
                        power_model,
                        _prepare_prediction_frame(scenario_raw_df, addon, target),
                        power_feat_cols,
                    )
                    q_demand_pred = None
                    q_required_min = None
                    q_capability = None
                    q_gap_pct = None
                    feasible = True
                    if q_demand_model is not None and q_demand_feat_cols:
                        q_frame = _prepare_q_demand_frame(scenario_raw_df, addon)
                        if q_frame.empty:
                            feasible = False
                        else:
                            missing_q_cols = [col for col in q_demand_feat_cols if col not in q_frame.columns]
                            if missing_q_cols:
                                feasible = False
                            else:
                                q_row = q_frame.iloc[[-1]][q_demand_feat_cols]
                                q_demand_pred = float(q_demand_model.predict(q_row)[0])
                                q_required_min = q_demand_pred * 1.05
                                q_capability = addon.decision.estimate_q_capability(scenario_raw_df, action)
                                if q_capability is None or q_required_min is None or q_required_min <= 0:
                                    feasible = False
                                else:
                                    q_gap_pct = (float(q_capability) - float(q_required_min)) / float(q_required_min) * 100.0
                                    feasible = bool(q_capability >= q_required_min)
                    score = {
                        "predicted_power": float(predicted_power),
                        "q_demand_pred": q_demand_pred,
                        "q_required_min": q_required_min,
                        "q_capability": q_capability,
                        "q_gap_pct": q_gap_pct,
                        "feasible": feasible,
                        "action": action,
                    }
                except Exception:
                    continue

                if fallback_score is None or score["predicted_power"] < fallback_score["predicted_power"]:
                    fallback_score = score
                if score["feasible"] and (best_score is None or score["predicted_power"] < best_score["predicted_power"]):
                    best_score = score

            chosen = best_score or fallback_score
            if chosen is None:
                continue

            projected_row = template_row.copy()
            simulated = addon.decision.simulate_dynamics(projected_history.iloc[-1], chosen["action"], chosen["q_demand_pred"])
            for key, value in simulated.items():
                projected_row[key] = value
            projected_row[target] = chosen["predicted_power"]
            projected_history = pd.concat([projected_history, pd.DataFrame([projected_row], index=[next_ts])])

            baseline_power = float(template_row.get(target, 0.0) or 0.0)
            optimized_power = max(float(chosen["predicted_power"]), 0.0)
            rows.append(
                {
                    "timestamp": next_ts.isoformat(),
                    "baseline_power_kw": baseline_power,
                    "optimized_power_kw": optimized_power,
                    "saving_power_kw": max(baseline_power - optimized_power, 0.0),
                    "q_required_min": chosen["q_required_min"],
                    "q_capability": chosen["q_capability"],
                    "q_gap_pct": chosen["q_gap_pct"],
                    "feasible": bool(chosen["feasible"]),
                }
            )
            if total_steps > 0:
                percent = 84 + int(((idx + 1) / total_steps) * 8)
                report_progress(
                    percent=min(percent, 91),
                    stage="future_projection",
                    message="正在整理未來一倍時長的預測趨勢...",
                    completed_projection_points=idx + 1,
                    total_projection_points=total_steps,
                )

        curve_df = pd.DataFrame(rows)
        baseline_kwh = float(curve_df["baseline_power_kw"].sum() * interval_h) if not curve_df.empty else 0.0
        optimized_kwh = float(curve_df["optimized_power_kw"].sum() * interval_h) if not curve_df.empty else 0.0
        saving_kwh = max(baseline_kwh - optimized_kwh, 0.0)
        return {
            "summary": {
                "baseline_kwh": round(baseline_kwh, 2),
                "optimized_kwh": round(optimized_kwh, 2),
                "saving_kwh": round(saving_kwh, 2),
                "saving_pct": round(saving_kwh / baseline_kwh * 100.0, 2) if baseline_kwh > 0 else 0.0,
                "rows_projected": int(len(curve_df)),
                "interval_hours": round(interval_h * len(curve_df), 2),
            },
            "curve": rows,
        }

    @classmethod
    def _position_within_bounds(cls, value: float, bounds: list[float]) -> float:
        lo = float(min(bounds))
        hi = float(max(bounds))
        if hi <= lo:
            return 0.5
        return round((value - lo) / (hi - lo), 3)
