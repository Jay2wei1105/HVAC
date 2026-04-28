from .addon_base import BaseDomainAddon
from .types import CleanResult, CleaningConfig
from . import ingestion, time_index, mapper, quality_engine, interpolation, report, contracts
import pandas as pd
from typing import Optional, List

def run_pipeline(
    csv_path: str,
    addon: BaseDomainAddon,
    timestep_minutes: int = 15,
    override_cleaning_config: Optional[CleaningConfig] = None,
    enabled_validators: Optional[List[str]] = None,
    compute_derived_features: bool = True,
) -> CleanResult:
    """
    通用 9 步驟 pipeline — 無視領域。
    所有領域邏輯透過 addon 介面注入。
    """
    # 1. Ingestion — 純讀檔
    df = ingestion.load_csv(csv_path)
    raw_rows = len(df)
    preclean = contracts.preclean_raw_frame(df)
    df = preclean.df
    
    # 2. Schema discovery — 問 add-on 要 sensor 定義
    schema = addon.get_sensor_schema()
    
    # 3. Column mapping — Layer 1 關鍵字 + Layer 2 數值特徵
    mapping_result = mapper.auto_map(df, schema)
    col_map = mapping_result.mapping
    df = mapper.apply_mapping(df, col_map)
    mapper.assert_required_columns(df, schema)  # 必要欄位檢查
    contract_result = contracts.coerce_and_validate_sensor_frame(df, schema)
    df = contract_result.df
    
    # 4. Time index — 解析時間欄、重建等距索引
    df = time_index.build(df, timestep_minutes)
    time_contract_issues = contracts.validate_time_index(df)
    
    # 5. Shutdown removal — 依 add-on 指定的欄位移除停機時段
    cleaning_cfg = override_cleaning_config if override_cleaning_config else addon.get_cleaning_config()
    df, shutdown_removed = quality_engine.remove_shutdown(df, cleaning_cfg)
    
    # 6. Physical range check — 依 sensor schema 的 physical_range
    df, range_violations = quality_engine.check_physical_range(df, schema)
    
    # 7. Spike & Flatline — 通用演算法（3σ / 連續不變）
    df, spike_count, flatline_count = quality_engine.detect_anomalies(df, cleaning_cfg)
    
    # 8. Cross validation — 呼叫 add-on 的物理檢核（能量守恆等）
    cross_issues = []
    validators = addon.get_cross_validators()
    if enabled_validators is not None:
        validators = [v for v in validators if v.__name__ in enabled_validators]
        
    for validator in validators:
        cross_issues.extend(validator(df))
    
    # 9. Interpolation — 通用插值（max_gap 由 add-on 決定）
    df, interp_count = interpolation.fill_short_gaps(df, cleaning_cfg)
    
    # Post: 衍生特徵（add-on 計算）
    if compute_derived_features:
        df = addon.compute_derived_features(df)
    
    # 輸出品質報告
    qr = report.build(
        df=df, addon_id=addon.domain_id,
        raw_rows=raw_rows,
        shutdown_removed=shutdown_removed,
        range_violations=range_violations,
        spike_count=spike_count,
        flatline_count=flatline_count,
        interpolated_count=interp_count,
        cross_issues=cross_issues,
        contract_issues=contract_result.issues + time_contract_issues,
        cleaning_actions=preclean.stats | contract_result.stats,
    )
    
    return CleanResult(df=df, quality_report=qr, column_mapping=col_map, addon_id=addon.domain_id)
