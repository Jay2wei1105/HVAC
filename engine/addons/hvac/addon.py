from engine.core.addon_base import BaseDomainAddon
from engine.core.types import CleaningConfig
from .sensors import HVAC_SENSOR_SCHEMA
from .equipment import HVAC_EQUIPMENT
from .validators import HVAC_VALIDATORS
from .features import compute_hvac_features

from .decision_part import HVACDecisionPart
from .dashboard_part import HVACDashboardPart
from .prediction_part import HVACPredictionPart
from .q_demand_part import HVACQDemandPart

class HVACAddon(BaseDomainAddon):
    domain_id = "hvac"
    display_name = "空調系統"
    version = "0.1.0"
    prediction = HVACPredictionPart()
    q_demand = HVACQDemandPart()
    decision = HVACDecisionPart()
    dashboard = HVACDashboardPart()
    
    def get_sensor_schema(self): return HVAC_SENSOR_SCHEMA
    def get_equipment_topology(self): return HVAC_EQUIPMENT
    def get_cross_validators(self): return HVAC_VALIDATORS
    def compute_derived_features(self, df): return compute_hvac_features(df)
    def get_cleaning_config(self):
        return CleaningConfig(
            spike_sigma=3.0,
            flatline_minutes=30,
            max_gap_minutes=60,
            shutdown_power_threshold=10.0,
            shutdown_detect_column="total_power",
        )
