from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .optimizer import run_control_optimization


@dataclass
class MPCSimulationResult:
    control_log: pd.DataFrame
    total_energy_mpc: float
    total_energy_oracle: float
    total_energy_baseline: float
    saving_ratio_vs_oracle: float
    passed_ratio_gate: bool
    summary: dict[str, float | bool] = field(default_factory=dict)


def simulate_dynamics(
    addon: Any,
    state_row: pd.Series,
    action: dict[str, float],
    q_demand_pred: float | None = None,
) -> pd.Series:
    if addon.decision is None:
        raise ValueError("addon.decision is required for MPC dynamics.")
    return pd.Series(addon.decision.simulate_dynamics(state_row, action, q_demand_pred))


def _clip_to_control_bounds(
    action: dict[str, float],
    control_vars: list[Any],
) -> dict[str, float]:
    bounds = {cv.name: cv.effective_bounds for cv in control_vars}
    clipped = dict(action)
    for name, value in action.items():
        if name not in bounds:
            continue
        lo, hi = bounds[name]
        clipped[name] = float(np.clip(value, lo, hi))
    return clipped


def _ensure_q_safety(
    addon: Any,
    scenario_df: pd.DataFrame,
    action: dict[str, float],
    control_vars: list[Any],
    q_required_min: float | None,
) -> tuple[dict[str, float], float | None, bool]:
    safe_action = _clip_to_control_bounds(action, control_vars)
    q_capability = addon.decision.estimate_q_capability(scenario_df, safe_action)
    if q_required_min is None:
        return safe_action, q_capability, True

    bounds = {cv.name: cv.effective_bounds for cv in control_vars}
    for _ in range(12):
        if q_capability is not None and q_capability >= q_required_min:
            return safe_action, q_capability, True

        changed = False
        if "chw_supply_temp" in safe_action and "chw_supply_temp" in bounds:
            lo, hi = bounds["chw_supply_temp"]
            next_val = float(np.clip(safe_action["chw_supply_temp"] - 0.5, lo, hi))
            changed = changed or next_val != safe_action["chw_supply_temp"]
            safe_action["chw_supply_temp"] = next_val
        if "chwp_freq" in safe_action and "chwp_freq" in bounds:
            lo, hi = bounds["chwp_freq"]
            next_val = float(np.clip(safe_action["chwp_freq"] + 5.0, lo, hi))
            changed = changed or next_val != safe_action["chwp_freq"]
            safe_action["chwp_freq"] = next_val

        q_capability = addon.decision.estimate_q_capability(scenario_df, safe_action)
        if not changed:
            break

    feasible = q_capability is not None and q_capability >= q_required_min
    return safe_action, q_capability, feasible


def run_mpc(
    raw_df: pd.DataFrame,
    addon: Any,
    power_model: Any,
    power_feat_cols: list[str],
    target: str,
    control_vars: list[Any],
    weights: Any,
    q_demand_model: Any | None = None,
    q_demand_feat_cols: list[str] | None = None,
    horizon_steps: int = 8,
    advisory_trials: int = 20,
    saving_ratio_threshold: float = 0.85,
) -> MPCSimulationResult:
    if addon.decision is None:
        raise ValueError("addon.decision is required for MPC.")

    working_df = raw_df.copy()
    working_df.index = pd.to_datetime(working_df.index, errors="coerce")
    working_df = working_df[working_df.index.notnull()].sort_index()
    if len(working_df) < max(32, horizon_steps + 4):
        raise ValueError("Not enough rows for MPC simulation.")

    executed_rows: list[dict[str, Any]] = []
    previous_action = {cv.name: cv.current_value for cv in control_vars}
    total_energy_mpc = 0.0
    total_energy_oracle = 0.0
    total_energy_baseline = 0.0

    for step_idx in range(min(len(working_df) - 1, horizon_steps)):
        history = working_df.iloc[: len(working_df) - horizon_steps + step_idx]
        advisory = run_control_optimization(
            model=power_model,
            feature_df=history,
            feat_cols=power_feat_cols,
            target=target,
            control_vars=control_vars,
            weights=weights,
            n_trials=advisory_trials,
            addon=addon,
            q_demand_model=q_demand_model,
            q_demand_feat_cols=q_demand_feat_cols,
        )
        proposed_action = advisory.best_setpoints
        rate_limited_action = addon.decision.apply_rate_limits(previous_action, proposed_action)

        scenario_df = pd.concat([history, pd.DataFrame([history.iloc[-1]], index=[history.index[-1]])])
        q_required_min = advisory.q_required_min
        safe_action, q_capability, feasible = _ensure_q_safety(
            addon,
            scenario_df,
            rate_limited_action,
            control_vars,
            q_required_min,
        )

        current_row = history.iloc[-1]
        next_state = simulate_dynamics(addon, current_row, safe_action, advisory.q_demand_pred)
        interval_h = 0.25

        total_energy_baseline += float(current_row.get(target, 0.0)) * interval_h
        total_energy_oracle += advisory.predicted_power * interval_h
        total_energy_mpc += float(next_state.get(target, advisory.predicted_power)) * interval_h

        executed_rows.append(
            {
                "timestamp": history.index[-1],
                "proposed_action": proposed_action,
                "executed_action": safe_action,
                "q_demand_pred": advisory.q_demand_pred,
                "q_required_min": q_required_min,
                "q_capability": q_capability,
                "feasible": feasible,
                "baseline_power": float(current_row.get(target, 0.0)),
                "oracle_power": advisory.predicted_power,
                "mpc_power": float(next_state.get(target, advisory.predicted_power)),
            }
        )
        previous_action = safe_action

    oracle_saving = max(total_energy_baseline - total_energy_oracle, 0.0)
    mpc_saving = max(total_energy_baseline - total_energy_mpc, 0.0)
    saving_ratio = 1.0 if oracle_saving == 0 else mpc_saving / oracle_saving

    log_df = pd.DataFrame(executed_rows)
    summary = {
        "total_energy_mpc": total_energy_mpc,
        "total_energy_oracle": total_energy_oracle,
        "total_energy_baseline": total_energy_baseline,
        "saving_ratio_vs_oracle": saving_ratio,
        "passed_ratio_gate": saving_ratio >= saving_ratio_threshold,
    }
    return MPCSimulationResult(
        control_log=log_df,
        total_energy_mpc=total_energy_mpc,
        total_energy_oracle=total_energy_oracle,
        total_energy_baseline=total_energy_baseline,
        saving_ratio_vs_oracle=saving_ratio,
        passed_ratio_gate=saving_ratio >= saving_ratio_threshold,
        summary=summary,
    )
