from engine.core.types import SensorDefinition, SensorType, SensorRole

HVAC_SENSOR_SCHEMA: list[SensorDefinition] = [
    # --- 冰水側 ---
    SensorDefinition("chw_supply_temp", SensorType.TEMPERATURE, SensorRole.CONTROLLABLE,
                     "°C", ["CHWS", "chw_supply", "冰水出水"], (3, 15), required=True,
                     value_hint=(5, 12)),
    SensorDefinition("chw_return_temp", SensorType.TEMPERATURE, SensorRole.OBSERVABLE,
                     "°C", ["CHWR", "chw_return", "冰水回水"], (5, 20), required=True),
    SensorDefinition("chw_flow_lpm", SensorType.FLOW, SensorRole.OBSERVABLE,
                     "L/min", ["CHW_Flow", "chw_flow_lpm", "冰水流量"], (0, 5000),
                     required=False, value_hint=(50, 4000)),
    # --- 冷卻水側 ---
    SensorDefinition("cw_supply_temp", SensorType.TEMPERATURE, SensorRole.OBSERVABLE,
                     "°C", ["CWS", "cw_supply", "冷卻水出水"], (20, 40)),
    SensorDefinition("cw_return_temp", SensorType.TEMPERATURE, SensorRole.OBSERVABLE,
                     "°C", ["CWR", "cw_return", "冷卻水回水"], (25, 45)),
    # --- 外氣 ---
    SensorDefinition("outdoor_temp", SensorType.TEMPERATURE, SensorRole.OBSERVABLE,
                     "°C", ["OA_Temp", "outdoor", "外氣溫"], (-5, 45), required=True),
    SensorDefinition("outdoor_rh", SensorType.HUMIDITY, SensorRole.OBSERVABLE,
                     "%", ["RH", "humidity", "濕度"], (0, 100)),
    # --- 功率 ---
    SensorDefinition("total_power", SensorType.POWER, SensorRole.OUTPUT,
                     "kW", ["SYS_kW", "Total_kW", "系統總功率"], (0, 5000), required=True),
    SensorDefinition("ch_kw", SensorType.POWER, SensorRole.OUTPUT,
                     "kW", ["CH_kW", "chiller_kw", "主機功率"], (0, 3000)),
    SensorDefinition("chwp_kw", SensorType.POWER, SensorRole.OUTPUT,
                     "kW", ["CHWP_kW", "冰水泵功率"], (0, 200)),
    SensorDefinition("cwp_kw", SensorType.POWER, SensorRole.OUTPUT,
                     "kW", ["CWP_kW", "冷卻水泵功率"], (0, 200)),
    SensorDefinition("ct_kw", SensorType.POWER, SensorRole.OUTPUT,
                     "kW", ["CT_kW", "冷卻塔功率"], (0, 100)),
    # --- 可控頻率 ---
    SensorDefinition("chwp_freq", SensorType.FREQUENCY, SensorRole.CONTROLLABLE,
                     "Hz", ["CHWP_Hz", "冰水泵頻率"], (25, 60)),
    SensorDefinition("cwp_freq", SensorType.FREQUENCY, SensorRole.CONTROLLABLE,
                     "Hz", ["CWP_Hz", "冷卻水泵頻率"], (25, 60)),
    SensorDefinition("ct_freq", SensorType.FREQUENCY, SensorRole.CONTROLLABLE,
                     "Hz", ["CT_Hz", "冷卻塔頻率"], (25, 60)),
    SensorDefinition("chiller_count", SensorType.COUNT, SensorRole.CONTROLLABLE,
                     "台", ["CH_Count", "主機台數"], (0, 10)),
]
