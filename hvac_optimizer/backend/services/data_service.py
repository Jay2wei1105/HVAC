import pandas as pd
import re

class DataService:
    @staticmethod
    def suggest_mappings(columns: list):
        suggestions = []
        
        comp_rules = {
            "CHWP": ["chw_p", "chwp", "chw", "冰水泵", "冰水側"], # CHW 放在這
            "CWP": ["cw_p", "cwp", "cw", "冷卻水泵", "冷卻水側"],
            "CHILLER": ["chiller", "冰機", "主機", "ch"], 
            "CT": ["ct", "tower", "水塔", "風扇", "fan"],
            "OA": ["oa", "ambient", "外氣", "室外", "環境"],
            "TIME": ["timestamp", "ts", "time", "時間", "日期"]
        }
        
        meas_rules = {
            "POWER": ["power", "kw", "電力", "功率", "耗電"],
            "TEMP": ["temp", "溫度", "溫", "t"],
            "RH": ["rh", "humidity" ,"濕度", "溼度"],
            "FREQ": ["freq", "hz", "頻率", "頻"],
            "FLOW": ["flow", "流量", "gpm", "lpm", "cms"],
            "SETPOINT": ["setp", "sp", "設定", "目標"]
        }

        for col in columns:
            col_str = str(col).lower()
            clean_col = re.sub(r'[^a-z0-9\u4e00-\u9fa5_]', '', col_str)
            target = None
            
            found_comp = None
            for comp, kws in comp_rules.items():
                if any(kw in clean_col for kw in kws):
                    found_comp = comp
                    break
            
            found_meas = None
            for meas, kws in meas_rules.items():
                if any(kw in clean_col for kw in kws):
                    found_meas = meas
                    break

            found_dir = None
            if any(k in clean_col for k in ["supply", "s", "供", "出"]): found_dir = "S"
            if any(k in clean_col for k in ["return", "r", "回"]): found_dir = "R"
            
            # --- 強化後的溫度決策邏輯 ---
            if found_comp == "TIME": target = "timestamp"
            elif "total" in clean_col or "sum" in clean_col or "總" in clean_col:
                if found_meas == "POWER": target = "total_power"
            elif found_comp == "OA":
                if found_meas == "TEMP": target = "ambient_temp"
                elif found_meas == "RH": target = "oa_rh"
            elif found_comp in ["CHILLER", "CHWP"]:
                # 冰機與冰水側共用溫度判斷 (由方向決定)
                if found_meas == "TEMP":
                    if found_dir == "R": target = "chwr_temp"
                    else: target = "chws_temp"
                elif found_comp == "CHILLER":
                    if found_meas == "POWER": target = "chiller_power"
                    elif found_meas == "FREQ": target = "ch_freq"
                    elif found_meas == "SETPOINT": target = "chws_setpoint"
                elif found_comp == "CHWP":
                    if found_meas == "POWER": target = "chwp_power"
                    elif found_meas == "FREQ": target = "chwp_freq"
                    elif found_meas == "FLOW": target = "chw_flow"
            elif found_comp in ["CWP", "CT"]:
                # 冷卻水與水塔側共用溫度判斷
                if found_meas == "TEMP":
                    if found_dir == "R": target = "cwr_temp"
                    else: target = "cws_temp"
                elif found_comp == "CWP":
                    if found_meas == "POWER": target = "cwp_power"
                    elif found_meas == "FREQ": target = "cwp_freq"
                    elif found_meas == "FLOW": target = "cw_flow"
                elif found_comp == "CT":
                    if found_meas == "POWER": target = "ct_fan_power"
                    elif found_meas == "FREQ": target = "ct_fan_freq"
            elif found_meas == "FLOW":
                if "chw" in clean_col: target = "chw_flow"
                elif "cw" in clean_col: target = "cw_flow"

            suggestions.append({
                "source": col, "target": target, "confidence": "high" if target else "low",
                "reason": f"智慧識別: {found_comp}+{found_meas}" if target else "建議手動選配"
            })
            
        return suggestions

    @staticmethod
    def suggest_equipment(mappings: list[dict]) -> dict:
        """
        Infer equipment families/counts from mapped targets and source names.
        Returns editable default specs for frontend onboarding.
        """

        def _infer_count(rows: list[dict]) -> int:
            if not rows:
                return 0
            ids = set()
            for row in rows:
                src = str(row.get("source", ""))
                # Only treat explicit device numbering as count evidence:
                # CHWP_1_*, CWP-2-*, CT 3 *, #1 ...
                found = re.findall(r"(?:^|[_\-\s#])(\d{1,2})(?:$|[_\-\s])", src)
                if found:
                    ids.add(int(found[-1]))
            if ids:
                return max(ids)
            # No explicit numbering: conservative default is single equipment,
            # because one device often contributes multiple columns (power/freq/flow).
            return 1

        by_target = {}
        for row in mappings:
            tgt = row.get("target")
            if not tgt:
                continue
            by_target.setdefault(tgt, []).append(row)

        chiller_rows = by_target.get("chiller_power", []) + by_target.get("ch_freq", [])
        chwp_rows = by_target.get("chwp_power", []) + by_target.get("chwp_freq", []) + by_target.get("chw_flow", [])
        cwp_rows = by_target.get("cwp_power", []) + by_target.get("cwp_freq", []) + by_target.get("cw_flow", [])
        ct_rows = by_target.get("ct_fan_power", []) + by_target.get("ct_fan_freq", [])

        chiller_count = _infer_count(chiller_rows)
        chwp_count = _infer_count(chwp_rows)
        cwp_count = _infer_count(cwp_rows)
        ct_count = _infer_count(ct_rows)

        if chiller_count == 0 and by_target.get("total_power"):
            chiller_count = 1

        return {
            "chillers": [
                {"id": idx + 1, "rt": 500.0, "cop": 5.5}
                for idx in range(chiller_count)
            ],
            "chwp": [
                {"id": idx + 1, "kw": 22.0, "freq_max_hz": 60.0}
                for idx in range(chwp_count)
            ],
            "cwp": [
                {"id": idx + 1, "kw": 22.0, "freq_max_hz": 60.0}
                for idx in range(cwp_count)
            ],
            "cooling_tower": [
                {"id": idx + 1, "kw": 18.5, "freq_max_hz": 60.0}
                for idx in range(ct_count)
            ],
            "detected": {
                "chillers": chiller_count,
                "chwp": chwp_count,
                "cwp": cwp_count,
                "cooling_tower": ct_count,
            },
        }
