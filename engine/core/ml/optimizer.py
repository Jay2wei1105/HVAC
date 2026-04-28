"""
Control Optimization Engine / Stage 3.

Uses the trained power model as the objective and optionally enforces an HVAC
cooling-capability constraint using a q-demand quantile model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from engine.core.ml.features_common import build_features_from_registry


@dataclass
class ControlVariable:
    """A single controllable variable with 3-layer bounds."""

    name: str
    display_name: str
    unit: str
    l1_bounds: tuple[float, float]
    l2_bounds: tuple[float, float]
    l3_bounds: tuple[float, float]
    current_value: float = 0.0

    @property
    def effective_bounds(self) -> tuple[float, float]:
        lo = max(self.l1_bounds[0], self.l2_bounds[0], self.l3_bounds[0])
        hi = min(self.l1_bounds[1], self.l2_bounds[1], self.l3_bounds[1])
        return (lo, hi)


@dataclass
class BusinessWeights:
    """Trade-off weights between energy, comfort, longevity."""

    energy: float = 1.0
    comfort: float = 100.0
    longevity: float = 5.0
    label: str = "balanced"


WEIGHT_PRESETS: dict[str, BusinessWeights] = {
    "full_saving": BusinessWeights(10.0, 50.0, 2.0, "full_saving"),
    "balanced": BusinessWeights(1.0, 100.0, 5.0, "balanced"),
    "comfort": BusinessWeights(0.5, 500.0, 2.0, "comfort"),
}


@dataclass
class OptimizationResult:
    """Output from the control optimization."""

    best_setpoints: dict[str, float]
    predicted_power: float
    baseline_power: float
    saving_pct: float
    saving_kwh: float
    trials_df: pd.DataFrame
    baseline_setpoints: dict[str, float] = field(default_factory=dict)
    all_trials: list[dict[str, Any]] = field(default_factory=list)
    q_demand_pred: float | None = None
    q_required_min: float | None = None
    q_capability: float | None = None
    q_constraint_multiplier: float = 1.05
    feasible: bool = True
    feasible_trials: int = 0
    total_trials: int = 0

    def to_payload(self) -> dict[str, Any]:
        recommendations: list[str] = []
        if self.feasible:
            recommendations.append("Optimization result satisfies the cooling-capability safety constraint.")
        elif self.q_required_min is not None and self.q_capability is not None:
            recommendations.append(
                "Best trial still misses the cooling-capability constraint; narrow the search space or increase available flow/capacity."
            )
        if self.q_demand_pred is not None:
            recommendations.append(
                f"Q-demand P90 = {self.q_demand_pred:.1f} kW, required minimum with margin = {self.q_required_min:.1f} kW."
            )
        if self.q_capability is not None:
            recommendations.append(f"Estimated Q capability at the recommended setpoints = {self.q_capability:.1f} kW.")

        best_solution = {
            "energy_cost": self.predicted_power,
            "setpoints": self.best_setpoints,
            "comfort_penalty": float(self.trials_df["comfort_penalty"].min()) if not self.trials_df.empty else 0.0,
            "total_score": float(self.trials_df["objective"].min()) if not self.trials_df.empty else self.predicted_power,
            "q_demand_pred": self.q_demand_pred,
            "q_required_min": self.q_required_min,
            "q_capability": self.q_capability,
            "feasible": self.feasible,
        }
        baseline = {
            "energy_cost": self.baseline_power,
            "setpoints": self.baseline_setpoints,
        }
        return {
            "best_solution": best_solution,
            "baseline": baseline,
            "trials_df": self.trials_df,
            "recommendations": recommendations,
            "feasible_trials": self.feasible_trials,
            "total_trials": self.total_trials,
        }


def extract_control_variables(
    df: pd.DataFrame,
    sensor_schema: list,
    site_config: Optional[dict] = None,
) -> list[ControlVariable]:
    from engine.core.types import SensorRole

    controls: list[ControlVariable] = []
    for sensor in sensor_schema:
        if sensor.role != SensorRole.CONTROLLABLE:
            continue
        if sensor.std_name not in df.columns:
            continue

        series = df[sensor.std_name].dropna()
        if series.empty:
            continue

        l1 = list(sensor.physical_range)
        if site_config and sensor.std_name == "chws_setpoint":
            equip = site_config.get("equipment", {}).get("chiller", {})
            c_min = equip.get("chws_min")
            c_max = equip.get("chws_max")
            if c_min is not None:
                l1[0] = float(c_min)
            if c_max is not None:
                l1[1] = float(c_max)

        l1 = tuple(l1)
        p1 = float(series.quantile(0.01))
        p99 = float(series.quantile(0.99))
        l2 = (max(l1[0], p1), min(l1[1], p99))

        controls.append(
            ControlVariable(
                name=sensor.std_name,
                display_name=sensor.keywords[0] if sensor.keywords else sensor.std_name,
                unit=sensor.unit,
                l1_bounds=l1,
                l2_bounds=l2,
                l3_bounds=l2,
                current_value=float(series.median()),
            )
        )

    return controls


def estimate_runtime(n_rows: int, n_trials: int, n_cores: int = 4) -> str:
    secs = (n_rows * n_trials * 0.001) / n_cores
    if secs < 60:
        return f"{secs:.0f} sec"
    return f"{secs / 60:.1f} min"


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[out.index.notnull()].sort_index()
    return out


def _infer_step(index: pd.DatetimeIndex) -> pd.Timedelta:
    if len(index) < 2:
        return pd.Timedelta(minutes=15)
    diffs = pd.Series(index[1:] - index[:-1])
    diffs = diffs[diffs > pd.Timedelta(0)]
    if diffs.empty:
        return pd.Timedelta(minutes=15)
    return diffs.mode().iloc[0]


def _append_trial_row(
    raw_df: pd.DataFrame,
    control_vars: list[ControlVariable],
    trial_values: dict[str, float],
    history_rows: int = 192,
) -> pd.DataFrame:
    history = _ensure_datetime_index(raw_df).iloc[-history_rows:].copy()
    scenario_row = history.iloc[-1].copy()
    for cv in control_vars:
        if cv.name in scenario_row.index:
            scenario_row[cv.name] = trial_values.get(cv.name, cv.current_value)
    next_ts = history.index[-1] + _infer_step(history.index)
    scenario = pd.DataFrame([scenario_row], index=[next_ts])
    return pd.concat([history, scenario])


def _prepare_prediction_frame(
    raw_df: pd.DataFrame,
    addon: Any,
    target: str,
) -> pd.DataFrame:
    working = _ensure_datetime_index(raw_df)
    working = addon.compute_derived_features(working.copy())
    working, _ = build_features_from_registry(working, target)
    return addon.prediction.build_features(working, target=target)


def _prepare_q_demand_frame(raw_df: pd.DataFrame, addon: Any) -> pd.DataFrame:
    working = _ensure_datetime_index(raw_df)
    working = addon.compute_derived_features(working.copy())
    return addon.q_demand.build_features(working)


def _predict_from_last_row(model: Any, frame: pd.DataFrame, feat_cols: list[str]) -> float:
    if frame.empty:
        raise ValueError("Feature frame is empty after preprocessing; cannot optimize.")
    missing = [col for col in feat_cols if col not in frame.columns]
    if missing:
        missing_str = ", ".join(missing[:5])
        raise ValueError(f"Feature frame is missing required columns: {missing_str}")
    row = frame.iloc[[-1]][feat_cols]
    return float(np.asarray(model.predict(row), dtype=float)[0])


def run_control_optimization(
    model: Any,
    feature_df: pd.DataFrame,
    feat_cols: list[str],
    target: str,
    control_vars: list[ControlVariable],
    weights: BusinessWeights,
    n_trials: int = 100,
    n_baseline_samples: int = 100,
    addon: Any | None = None,
    q_demand_model: Any | None = None,
    q_demand_feat_cols: list[str] | None = None,
    q_constraint_multiplier: float = 1.05,
) -> OptimizationResult:
    try:
        import optuna
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Optuna is required for control optimization. Install with: pip install optuna"
        ) from exc

    if addon is None:
        raise ValueError("addon is required so optimization can rebuild consistent model features.")
    if addon.decision is None:
        raise ValueError("addon.decision is required for Stage 3 optimization.")

    base_raw_df = _ensure_datetime_index(feature_df)
    current_main_frame = _prepare_prediction_frame(base_raw_df, addon, target)
    baseline_power = _predict_from_last_row(model, current_main_frame, feat_cols)
    baseline_setpoints = {cv.name: cv.current_value for cv in control_vars}
    comfort_ref = baseline_setpoints.copy()

    all_trials_data: list[dict[str, Any]] = []
    feasible_trials = 0

    def objective(trial: Any) -> float:
        nonlocal feasible_trials

        trial_vals: dict[str, float] = {}
        for cv in control_vars:
            lo, hi = cv.effective_bounds
            if lo >= hi:
                trial_vals[cv.name] = lo
            else:
                trial_vals[cv.name] = trial.suggest_float(cv.name, lo, hi)

        scenario_raw_df = _append_trial_row(base_raw_df, control_vars, trial_vals)
        comfort_penalty = sum(abs(trial_vals.get(k, 0.0) - v) for k, v in comfort_ref.items())
        longevity_penalty = 0.0

        q_demand_pred = None
        q_required_min = None
        q_capability = None
        feasible = True
        constraint_penalty = 0.0
        pred_power = baseline_power
        error_message = None

        try:
            main_frame = _prepare_prediction_frame(scenario_raw_df, addon, target)
            pred_power = _predict_from_last_row(model, main_frame, feat_cols)
        except (KeyError, ValueError) as exc:
            feasible = False
            constraint_penalty = max(constraint_penalty, 2_000_000.0)
            error_message = str(exc)

        if feasible and q_demand_model is not None and q_demand_feat_cols:
            try:
                q_frame = _prepare_q_demand_frame(scenario_raw_df, addon)
                if q_frame.empty:
                    feasible = False
                    constraint_penalty = max(constraint_penalty, 1_500_000.0)
                    error_message = "Q-demand feature frame is empty after preprocessing."
                else:
                    missing_q_cols = [col for col in q_demand_feat_cols if col not in q_frame.columns]
                    if missing_q_cols:
                        feasible = False
                        constraint_penalty = max(constraint_penalty, 1_500_000.0)
                        error_message = (
                            "Q-demand feature frame is missing required columns: "
                            + ", ".join(missing_q_cols[:5])
                        )
                    else:
                        q_row = q_frame.iloc[[-1]][q_demand_feat_cols]
                        q_demand_pred = float(np.asarray(q_demand_model.predict(q_row), dtype=float)[0])
                        q_required_min = q_demand_pred * q_constraint_multiplier
                        q_capability = addon.decision.estimate_q_capability(scenario_raw_df, trial_vals)
                        if q_capability is None:
                            feasible = False
                            constraint_penalty = max(constraint_penalty, 1_000_000.0)
                            error_message = "Q capability could not be estimated for this trial."
                        elif q_capability < q_required_min:
                            feasible = False
                            deficit = q_required_min - q_capability
                            constraint_penalty = max(constraint_penalty, 1_000_000.0 + deficit * 10_000.0)
                            error_message = "Q capability is below the required safety margin."
            except (KeyError, ValueError) as exc:
                feasible = False
                constraint_penalty = max(constraint_penalty, 1_500_000.0)
                error_message = str(exc)

        if feasible:
            feasible_trials += 1

        obj = (
            weights.energy * pred_power
            + weights.comfort * comfort_penalty
            + weights.longevity * longevity_penalty
            + constraint_penalty
        )

        all_trials_data.append(
            {
                "trial": trial.number,
                "predicted_power": pred_power,
                "comfort_penalty": comfort_penalty,
                "longevity_penalty": longevity_penalty,
                "objective": obj,
                "energy_cost": pred_power,
                "total_score": obj,
                "q_demand_pred": q_demand_pred,
                "q_required_min": q_required_min,
                "q_capability": q_capability,
                "feasible": feasible,
                "error": error_message,
                **trial_vals,
            }
        )
        return obj

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    trials_df = pd.DataFrame(all_trials_data)
    candidate_trials = trials_df[trials_df["feasible"]].copy() if not trials_df.empty else pd.DataFrame()
    if candidate_trials.empty:
        candidate_trials = trials_df.copy()

    if candidate_trials.empty:
        best_vals = baseline_setpoints.copy()
        best_power = baseline_power
        best_feasible = False
        best_raw_df = _append_trial_row(base_raw_df, control_vars, best_vals)
    else:
        best_row = candidate_trials.sort_values("objective", ascending=True).iloc[0]
        best_vals = {cv.name: float(best_row.get(cv.name, cv.current_value)) for cv in control_vars}
        best_feasible = bool(best_row.get("feasible", False))
        best_raw_df = _append_trial_row(base_raw_df, control_vars, best_vals)
        try:
            best_main_frame = _prepare_prediction_frame(best_raw_df, addon, target)
            best_power = _predict_from_last_row(model, best_main_frame, feat_cols)
        except (KeyError, ValueError):
            best_vals = baseline_setpoints.copy()
            best_power = baseline_power
            best_feasible = False
            best_raw_df = _append_trial_row(base_raw_df, control_vars, best_vals)

    saving_pct = (baseline_power - best_power) / baseline_power if baseline_power > 0 else 0.0
    saving_kwh = baseline_power - best_power

    best_q_demand_pred = None
    best_q_required_min = None
    best_q_capability = None
    if q_demand_model is not None and q_demand_feat_cols:
        best_q_frame = _prepare_q_demand_frame(best_raw_df, addon)
        if not best_q_frame.empty:
            missing_q_cols = [col for col in q_demand_feat_cols if col not in best_q_frame.columns]
            if not missing_q_cols:
                q_row = best_q_frame.iloc[[-1]][q_demand_feat_cols]
                best_q_demand_pred = float(np.asarray(q_demand_model.predict(q_row), dtype=float)[0])
                best_q_required_min = best_q_demand_pred * q_constraint_multiplier
                best_q_capability = addon.decision.estimate_q_capability(best_raw_df, best_vals)
                if best_q_capability is None or best_q_capability < best_q_required_min:
                    best_feasible = False

    return OptimizationResult(
        best_setpoints=best_vals,
        predicted_power=best_power,
        baseline_power=baseline_power,
        saving_pct=saving_pct,
        saving_kwh=saving_kwh,
        trials_df=trials_df,
        baseline_setpoints=baseline_setpoints,
        all_trials=all_trials_data,
        q_demand_pred=best_q_demand_pred,
        q_required_min=best_q_required_min,
        q_capability=best_q_capability,
        q_constraint_multiplier=q_constraint_multiplier,
        feasible=best_feasible,
        feasible_trials=feasible_trials,
        total_trials=n_trials,
    )
