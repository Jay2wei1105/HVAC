from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from hvac_optimizer.backend.services.core_hvac_service import CoreHVACService


def _make_source_csv(path: Path, n: int = 220) -> None:
    index = pd.date_range("2024-01-01", periods=n, freq="15min")
    hour = index.hour.to_numpy()
    outdoor_temp = 28 + 4 * np.sin(2 * np.pi * hour / 24) + np.linspace(0, 1.0, n)
    outdoor_rh = 65 + 8 * np.cos(2 * np.pi * hour / 24)
    q = 420 + 10 * outdoor_temp + 0.2 * np.arange(n)
    total_power = 0.6 * q + 40
    df = pd.DataFrame(
        {
            "time": index,
            "CHWS": 7.0 + 0.15 * np.sin(2 * np.pi * np.arange(n) / 48),
            "CHWR": 12.0 + 0.2 * np.cos(2 * np.pi * np.arange(n) / 48),
            "CHW_Flow": 1000.0 + 20 * np.sin(2 * np.pi * np.arange(n) / 32),
            "CHWP_Hz": 50.0 + 1.5 * np.sin(2 * np.pi * np.arange(n) / 24),
            "CT_Hz": 42.0 + 1.0 * np.cos(2 * np.pi * np.arange(n) / 20),
            "CWS": 28.0 + 0.3 * np.sin(2 * np.pi * np.arange(n) / 36),
            "OA_Temp": outdoor_temp,
            "RH": outdoor_rh,
            "SYS_kW": total_power,
            "CH_kW": total_power * 0.7,
            "CHWP_kW": total_power * 0.08,
            "CWP_kW": total_power * 0.07,
            "CT_kW": total_power * 0.05,
        }
    )
    df.to_csv(path, index=False)


def test_hvac_optimizer_stage1_to_stage5_flow() -> None:
    storage = CoreHVACService.storage_dir()
    source = storage / "test_stage_flow.csv"
    artifact_root = storage / "site_test_core"
    if artifact_root.exists():
        shutil.rmtree(artifact_root)
    if source.exists():
        source.unlink()
    _make_source_csv(source)

    mappings = [
        {"source": "time", "target": "timestamp"},
        {"source": "CHWS", "target": "chws_temp"},
        {"source": "CHWR", "target": "chwr_temp"},
        {"source": "CHW_Flow", "target": "chw_flow"},
        {"source": "CHWP_Hz", "target": "chwp_freq"},
        {"source": "CT_Hz", "target": "ct_fan_freq"},
        {"source": "CWS", "target": "cws_temp"},
        {"source": "OA_Temp", "target": "ambient_temp"},
        {"source": "RH", "target": "oa_rh"},
        {"source": "SYS_kW", "target": "total_power"},
        {"source": "CH_kW", "target": "chiller_power"},
        {"source": "CHWP_kW", "target": "chwp_power"},
        {"source": "CWP_kW", "target": "cwp_power"},
        {"source": "CT_kW", "target": "ct_fan_power"},
    ]

    stage1 = CoreHVACService.run_stage1_pipeline(
        original_path=str(source),
        mappings=mappings,
        site_id="site_test_core",
        dataset_id="ds_test_core",
    )
    assert Path(stage1["cleaned_parquet_path"]).exists()
    assert stage1["quality_report"]["final_rows"] > 0

    ml = CoreHVACService.train_models(
        site_id="site_test_core",
        dataset_id="ds_test_core",
        cleaned_parquet_path=stage1["cleaned_parquet_path"],
        n_trials=2,
    )
    assert Path(ml["model_bundle_path"]).exists()
    assert "q_demand_metrics" in ml

    result = CoreHVACService.optimize_site(
        site_id="site_test_core",
        dataset_id="ds_test_core",
        cleaned_parquet_path=stage1["cleaned_parquet_path"],
        mappings=mappings,
        model_bundle_path=ml["model_bundle_path"],
        bounds={
            "chws": [6.0, 9.0],
            "chwp": [40.0, 55.0],
            "ct_fan": [35.0, 50.0],
            "cws": [26.0, 30.0],
        },
        quality_report=stage1["quality_report"],
        ml_results=ml,
    )
    assert result["q_constraint"]["q_capability"] is not None
    assert "mpc" in result
    assert Path(result["artifacts"]["optimization_results_parquet"]).exists()
    assert Path(result["artifacts"]["html_path"]).exists()
