from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, List, Dict, Tuple, Any
import pandas as pd

class SensorType(str, Enum):
    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    FLOW = "flow"
    POWER = "power"
    FREQUENCY = "frequency"
    HUMIDITY = "humidity"
    VOLTAGE = "voltage"
    CURRENT = "current"
    STATE = "state"           # on/off, 0/1
    COUNT = "count"           # 台數
    GENERIC_NUMERIC = "numeric"

class SensorRole(str, Enum):
    CONTROLLABLE = "controllable"    # 可控變量（頻率、設定點）
    OBSERVABLE = "observable"         # 只觀測（外氣、負荷）
    OUTPUT = "output"                 # 系統輸出（功率、COP）
    METADATA = "metadata"             # 輔助欄位

@dataclass
class SensorDefinition:
    std_name: str                     # 標準欄位名（add-on 定義）
    sensor_type: SensorType
    role: SensorRole
    unit: str
    keywords: list[str]               # CSV 欄位名比對關鍵字
    physical_range: tuple[float, float]  # (min, max) 物理合理範圍
    required: bool = False            # 缺此欄位則 pipeline 失敗
    value_hint: Optional[tuple[float, float]] = None  # 數值特徵（for Layer 2 mapper）

@dataclass
class EquipmentDefinition:
    equipment_id: str                 # 'chiller', 'chwp', 'compressor'...
    display_name: str
    sensor_refs: list[str]            # 關聯的 sensor std_name 列表
    metadata: dict = field(default_factory=dict)  # 額定功率、型號等

@dataclass
class CleaningConfig:
    # 由 add-on 提供，Core 執行
    spike_sigma: float = 3.0
    flatline_minutes: int = 30
    max_gap_minutes: int = 60
    shutdown_power_threshold: float = 10.0   # 小於此值視為停機（kW）
    shutdown_detect_column: Optional[str] = None  # 由哪個欄位判斷停機

@dataclass
class CleanResult:
    df: pd.DataFrame
    quality_report: "QualityReport"
    column_mapping: dict[str, str]    # std_name → 原始欄位名
    addon_id: str
