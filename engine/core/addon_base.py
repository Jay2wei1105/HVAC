from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Protocol, runtime_checkable

import pandas as pd

from .types import CleaningConfig, EquipmentDefinition, SensorDefinition


@runtime_checkable
class PredictionPart(Protocol):
    def get_prediction_targets(self) -> list[str]: ...
    def build_features(self, df: pd.DataFrame, target: str) -> pd.DataFrame: ...
    def get_feature_columns(self, df: pd.DataFrame, target: str) -> list[str]: ...
    def get_model_preset(self) -> dict: ...


@runtime_checkable
class QDemandPart(Protocol):
    def get_prediction_targets(self) -> list[str]: ...
    def build_features(self, df: pd.DataFrame, target: str = "q_delivered_kw") -> pd.DataFrame: ...
    def get_feature_columns(self, df: pd.DataFrame, target: str = "q_delivered_kw") -> list[str]: ...
    def get_model_preset(self) -> dict: ...


@runtime_checkable
class DecisionPart(Protocol):
    def get_rate_limits(self) -> dict[str, float]: ...
    def estimate_q_capability(
        self,
        scenario_df: pd.DataFrame,
        trial_values: dict[str, float],
    ) -> float | None: ...
    def apply_rate_limits(
        self,
        previous_action: dict[str, float] | None,
        proposed_action: dict[str, float],
    ) -> dict[str, float]: ...
    def simulate_dynamics(
        self,
        state_row: pd.Series,
        action: dict[str, float],
        q_demand_pred: float | None = None,
    ) -> dict[str, float]: ...


@runtime_checkable
class DashboardPart(Protocol):
    def build_dashboard_payload(
        self,
        *,
        quality_report: object | None = None,
        training_summary: dict | None = None,
        optimization_summary: dict | None = None,
        mpc_summary: dict | None = None,
        report_summary: dict | None = None,
    ) -> dict: ...


class BaseDomainAddon(ABC):
    prediction: PredictionPart
    q_demand: QDemandPart | None = None
    decision: DecisionPart | None = None
    dashboard: DashboardPart | None = None

    @property
    @abstractmethod
    def domain_id(self) -> str: ...

    @property
    @abstractmethod
    def display_name(self) -> str: ...

    @property
    @abstractmethod
    def version(self) -> str: ...

    @abstractmethod
    def get_sensor_schema(self) -> list[SensorDefinition]:
        """Return the standard sensor schema for this domain."""

    @abstractmethod
    def get_equipment_topology(self) -> list[EquipmentDefinition]:
        """Return the equipment graph / topology for this domain."""

    @abstractmethod
    def get_cleaning_config(self) -> CleaningConfig:
        """Return cleaning and interpolation rules."""

    @abstractmethod
    def get_cross_validators(self) -> list[Callable]:
        """Return cross-column validators; each validator accepts a DataFrame."""

    @abstractmethod
    def compute_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute domain-specific derived features after data cleaning."""
