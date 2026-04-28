from __future__ import annotations

import numpy as np
import pandas as pd

from .equipment import HVAC_EQUIPMENT

CP_WATER = 4.186
RHO_WATER = 1.0


class HVACDecisionPart:
    """HVAC-specific control constraints, capability estimation, and MPC dynamics."""

    def get_rate_limits(self) -> dict[str, float]:
        return {
            "chw_supply_temp": 0.5,
            "chwp_freq": 5.0,
            "cwp_freq": 5.0,
            "ct_freq": 5.0,
            "chiller_count": 1.0,
        }

    def estimate_q_capability(
        self,
        scenario_df: pd.DataFrame,
        trial_values: dict[str, float],
    ) -> float | None:
        row = scenario_df.iloc[-1]
        supply = float(trial_values.get("chw_supply_temp", row.get("chw_supply_temp", np.nan)))
        return_temp = float(row.get("chw_return_temp", np.nan))
        if np.isnan(supply) or np.isnan(return_temp):
            return None

        base_flow = row.get("chw_flow_lpm", np.nan)
        trial_freq = float(trial_values.get("chwp_freq", row.get("chwp_freq", np.nan)))
        base_freq = float(row.get("chwp_freq", np.nan))

        if pd.notna(base_flow):
            flow_lpm = float(base_flow)
            if pd.notna(base_freq) and base_freq > 0 and pd.notna(trial_freq):
                flow_lpm *= trial_freq / base_freq
        elif pd.notna(trial_freq):
            rated_lpm, rated_hz = self._chwp_rated_flow_hz()
            flow_lpm = rated_lpm * trial_freq / rated_hz
        else:
            return None

        delta_t = max(return_temp - supply, 0.0)
        q_cap = flow_lpm / 60.0 * RHO_WATER * CP_WATER * delta_t

        base_count = row.get("chiller_count", np.nan)
        trial_count = float(trial_values.get("chiller_count", base_count if pd.notna(base_count) else 1.0))
        if pd.notna(base_count) and float(base_count) > 0:
            q_cap *= max(trial_count, 0.0) / float(base_count)

        return float(max(q_cap, 0.0))

    def apply_rate_limits(
        self,
        previous_action: dict[str, float] | None,
        proposed_action: dict[str, float],
    ) -> dict[str, float]:
        if not previous_action:
            return dict(proposed_action)

        rate_limits = self.get_rate_limits()
        clipped = dict(proposed_action)
        for name, proposed in proposed_action.items():
            if name not in previous_action or name not in rate_limits:
                continue
            previous = float(previous_action[name])
            delta = float(rate_limits[name])
            clipped[name] = float(np.clip(proposed, previous - delta, previous + delta))
        return clipped

    def simulate_dynamics(
        self,
        state_row: pd.Series,
        action: dict[str, float],
        q_demand_pred: float | None = None,
    ) -> dict[str, float]:
        next_state = state_row.to_dict()

        if "chw_supply_temp" in action:
            current = float(state_row.get("chw_supply_temp", action["chw_supply_temp"]))
            target = float(action["chw_supply_temp"])
            next_state["chw_supply_temp"] = current + 0.7 * (target - current)

        if "chwp_freq" in action:
            current = float(state_row.get("chwp_freq", action["chwp_freq"]))
            target = float(action["chwp_freq"])
            next_state["chwp_freq"] = current + 0.8 * (target - current)

        if "chw_return_temp" in next_state and q_demand_pred is not None:
            supply = float(next_state.get("chw_supply_temp", state_row.get("chw_supply_temp", 7.0)))
            freq = float(next_state.get("chwp_freq", state_row.get("chwp_freq", 50.0)))
            flow_lpm = self._flow_from_freq(freq)
            if flow_lpm > 0:
                demand_dt = q_demand_pred * 60.0 / (flow_lpm * RHO_WATER * CP_WATER)
                next_state["chw_return_temp"] = supply + max(demand_dt, 0.0)

        return next_state

    @staticmethod
    def _chwp_rated_flow_hz() -> tuple[float, float]:
        for eq in HVAC_EQUIPMENT:
            if eq.equipment_id == "chwp":
                metadata = eq.metadata
                return float(metadata.get("rated_lpm", 630.0)), float(metadata.get("rated_hz", 50.0))
        return 630.0, 50.0

    def _flow_from_freq(self, freq_hz: float) -> float:
        rated_lpm, rated_hz = self._chwp_rated_flow_hz()
        return rated_lpm * float(freq_hz) / rated_hz if rated_hz > 0 else rated_lpm
