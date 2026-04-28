from __future__ import annotations

import pandas as pd

from engine.core.addon_base import BaseDomainAddon
from engine.core.types import CleaningConfig

from .prediction_part import CompressedAirPredictionPart
from .sensors import COMPRESSED_AIR_SENSOR_SCHEMA


class CompressedAirAddon(BaseDomainAddon):
    domain_id = "compressed_air"
    display_name = "Compressed Air"
    version = "0.1.0"
    prediction = CompressedAirPredictionPart()

    def get_sensor_schema(self):
        return COMPRESSED_AIR_SENSOR_SCHEMA

    def get_equipment_topology(self):
        return []

    def get_cross_validators(self):
        return []

    def compute_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.copy()

    def get_cleaning_config(self):
        return CleaningConfig(
            shutdown_power_threshold=5.0,
            shutdown_detect_column="total_power",
        )
