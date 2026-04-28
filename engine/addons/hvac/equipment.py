from engine.core.types import EquipmentDefinition

HVAC_EQUIPMENT = [
    EquipmentDefinition("chiller", "冰水主機",
        ["chw_supply_temp", "chw_return_temp", "cw_supply_temp", "cw_return_temp", "ch_kw", "chiller_count"],
        metadata={"rated_rt": 350, "rated_kw": 250, "count": 1}),
    EquipmentDefinition("chwp", "冰水泵",
        ["chwp_kw", "chwp_freq", "chw_flow_lpm"],
        metadata={"rated_kw": 15, "is_vfd": True, "rated_lpm": 630, "rated_hz": 50}),
    EquipmentDefinition("cwp", "冷卻水泵",
        ["cwp_kw", "cwp_freq"],
        metadata={"rated_kw": 18.5, "is_vfd": False}),
    EquipmentDefinition("cooling_tower", "冷卻塔",
        ["ct_kw", "ct_freq", "cw_supply_temp"],
        metadata={"rated_kw": 7.5, "is_vfd": False}),
]
