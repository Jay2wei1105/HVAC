import pandas as pd

def energy_balance_check(df: pd.DataFrame) -> list[dict]:
    """總功率 ≈ 各設備功率加總，誤差 > 5% 視為異常。"""
    issues = []
    components = ["ch_kw", "chwp_kw", "cwp_kw", "ct_kw"]
    if not all(c in df.columns for c in components + ["total_power"]):
        return issues
    summed = df[components].sum(axis=1)
    err = (df["total_power"] - summed).abs() / df["total_power"].replace(0, pd.NA)
    bad = (err > 0.05).fillna(False)
    if bad.any():
        issues.append({"rule": "energy_balance", "bad_rows": int(bad.sum()),
                       "hint": "子設備功率加總與總功率偏差 > 5%"})
    return issues

def chw_delta_t_physical(df: pd.DataFrame) -> list[dict]:
    """冰水回水 > 供水，ΔT 合理範圍 2~10°C。"""
    issues = []
    if "chw_supply_temp" not in df.columns or "chw_return_temp" not in df.columns:
        return issues
    dt = df["chw_return_temp"] - df["chw_supply_temp"]
    bad = ((dt < 1) | (dt > 12)).fillna(False)
    if bad.any():
        issues.append({"rule": "chw_delta_t", "bad_rows": int(bad.sum()),
                       "hint": "冰水 ΔT 超出 1~12°C 合理範圍"})
    return issues

HVAC_VALIDATORS = [energy_balance_check, chw_delta_t_physical]
