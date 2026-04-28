import pandas as pd
import numpy as np
import re

class DataService:
    @staticmethod
    def clean_hvac_data(df: pd.DataFrame):
        stats = {"initial_rows": len(df), "duplicates_removed": 0, "outliers_detected": 0, "nulls_filled": 0}
        initial_len = len(df)
        df = df.drop_duplicates()
        stats["duplicates_removed"] = int(initial_len - len(df))
        for col in df.columns:
            if str(col).lower() not in ['timestamp', 'time', 'ts', 'date'] and '時間' not in str(col):
                df[col] = pd.to_numeric(df[col], errors='coerce')
        stats["nulls_filled"] = int(df.isnull().sum().sum())
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if not numeric_cols.empty:
            df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit_direction='both')
        df = df.ffill().bfill()
        outlier_total = 0
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if any(k in str(col).lower() for k in ['power', 'kw', '電力']):
                mask = df[col] < 0
                outlier_total += int(mask.sum())
                df.loc[mask, col] = 0
            if any(k in str(col).lower() for k in ['temp', 'rt', '溫度']):
                mask = (df[col] > 100) | (df[col] < -20)
                outlier_total += int(mask.sum())
                if not df[col].empty:
                    df.loc[mask, col] = df[col].median()
        stats["outliers_detected"] = int(outlier_total)
        stats["final_rows"] = len(df)
        return df, stats

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
