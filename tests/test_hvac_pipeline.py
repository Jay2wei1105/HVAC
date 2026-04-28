import pytest
import pandas as pd
from engine.core.pipeline import run_pipeline
from engine.addons.hvac import HVACAddon

def test_hvac_pipeline_e2e():
    """Verify the HVAC pipeline runs end-to-end with mock data."""
    csv_path = "data/example.csv"
    addon = HVACAddon()
    
    result = run_pipeline(csv_path, addon, timestep_minutes=15)
    
    # 1. Verify CleanResult structure
    assert result.addon_id == "hvac"
    assert isinstance(result.df, pd.DataFrame)
    assert result.quality_report is not None
    
    # 2. Verify Column Mapping (from example.csv to std names)
    # CHWS_Temp -> chw_supply_temp
    # Total_kW -> total_power
    assert "chw_supply_temp" in result.df.columns
    assert "total_power" in result.df.columns
    
    # 3. Verify Quality Engine
    # Row at 01:30:00 had CHWS_Temp=50.0 (out of physical range 3-15)
    # Note: check_physical_range sets it to NaN
    # Since it's a single point, interpolation might fill it back if limit allows.
    # However, in our mock, 50.0 is definitely an anomaly.
    
    # Gap: 02:00 to 04:45 (12 points). limit=4. Points 03:00-03:45 should remain NaN.
    shutdown_row = result.df.loc["2024-01-01 03:30:00"]
    assert pd.isna(shutdown_row["total_power"])
    
    # 4. Verify Derived Features (B+: q_delivered_kw, is_steady)
    assert "chw_delta_t" in result.df.columns
    assert "q_delivered_kw" in result.df.columns
    assert "is_steady" in result.df.columns
    assert "system_kw_per_rt" in result.df.columns
    
    # 5. Verify Quality Report
    report = result.quality_report
    assert report.raw_rows == 22
    assert report.shutdown_removed == 12
    
def test_interchangeability_mock_addon():
    """Verify the pipeline runs with a minimal mock addon (Interchangeability test)."""
    from engine.core.addon_base import BaseDomainAddon
    from engine.core.types import SensorDefinition, SensorType, SensorRole, CleaningConfig
    
    class MockAddon(BaseDomainAddon):
        domain_id = "mock"
        display_name = "Mock System"
        version = "1.0.0"
        
        def get_sensor_schema(self):
            return [
                SensorDefinition("val_1", SensorType.GENERIC_NUMERIC, SensorRole.OBSERVABLE,
                                 "unit", ["Total_kW"], (0, 1000), required=True)
            ]
        def get_equipment_topology(self): return []
        def get_cleaning_config(self): return CleaningConfig()
        def get_cross_validators(self): return []
        def compute_derived_features(self, df): return df
        
    csv_path = "data/example.csv"
    result = run_pipeline(csv_path, MockAddon(), timestep_minutes=15)
    
    assert result.addon_id == "mock"
    assert "val_1" in result.df.columns
