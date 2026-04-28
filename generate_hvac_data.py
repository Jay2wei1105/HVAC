import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_hvac_dataset():
    """
    Generates a realistic 1-month HVAC dataset for summer, 
    including intentional data quality issues.
    """
    start_date = datetime(2024, 7, 1, 0, 0)
    end_date = start_date + timedelta(days=30)
    freq = "5min"
    
    timestamps = pd.date_range(start=start_date, end=end_date, freq=freq, inclusive='left')
    n = len(timestamps)
    
    # 1. Outdoor Environment (Diurnal Cycle)
    hour = timestamps.hour + timestamps.minute / 60
    # Temp: 28C at night, 35C at 14:00
    base_temp = 31.5 + 3.5 * np.sin((hour - 10) * np.pi / 12) 
    outdoor_temp = base_temp + np.random.normal(0, 0.5, n)
    # Humidity: Inverse of temp, 60% - 85%
    outdoor_rh = 75 - 10 * np.sin((hour - 10) * np.pi / 12) + np.random.normal(0, 2, n)
    
    # 2. System Load (Occupancy driven for office building)
    # Load peak at 14:00, off at night (keep some base load)
    load_factor_series = np.clip(0.4 + 0.5 * np.sin((hour - 9) * np.pi / 11), 0.1, 1.0)
    load_factor = load_factor_series.to_numpy()
    # Weekend reduction
    is_weekend = timestamps.weekday >= 5
    load_factor[is_weekend] *= 0.3
    
    # 3. Chilled Water Side (Primary Loop)
    chw_supply_temp = 7.0 + np.random.normal(0, 0.1, n)
    # Return temp: Supply + DeltaT. DeltaT = Load / (Flow * factor)
    # Assume flow is mostly constant but slightly modulated by freq
    chwp_freq = 50 + 5 * load_factor + np.random.normal(0, 0.5, n)
    chw_flow = 500 * (chwp_freq / 60) # m3/h
    # Delta T: 2C - 6C
    chw_delta_t = 5.0 * load_factor + np.random.normal(0, 0.2, n)
    chw_return_temp = chw_supply_temp + chw_delta_t
    
    # 4. Cooling Water Side
    # Return depends on heat rejection (Chiller Power + Cooling Load)
    cw_supply_temp = outdoor_temp * 0.7 + 5 + np.random.normal(0, 0.2, n) # Approach
    cw_delta_t = 4.5 * load_factor + np.random.normal(0, 0.2, n)
    cw_return_temp = cw_supply_temp + cw_delta_t
    cwp_freq = 55 + 5 * load_factor + np.random.normal(0, 0.2, n)
    ct_freq = 40 + 20 * load_factor + np.random.normal(0, 0.5, n)
    
    # 5. Power Consumption
    # Chiller: RT = Flow * DeltaT * constant
    cooling_rt = (chw_flow * chw_delta_t) / 3.024 # kcal/h to RT approx
    chiller_kw = cooling_rt * (0.6 + 0.1 * (1-load_factor)) # kW/RT decreases with efficiency
    
    # Pumps & Fans: Cubic law
    chwp_kw = 15 * (chwp_freq / 60)**3
    cwp_kw = 18.5 * (cwp_freq / 60)**3
    ct_kw = 7.5 * (ct_freq / 60)**3
    
    total_power = chiller_kw + chwp_kw + cwp_kw + ct_kw + 2.0 # + miscellaneous
    
    # --- INSERT DATA QUALITY ISSUES ---
    df = pd.DataFrame({
        "ts": timestamps,
        "OA_Temp": outdoor_temp,
        "OA_RH": outdoor_rh,
        "CHW_Flow": chw_flow,
        "CHW_Supply_Temp": chw_supply_temp,
        "CHW_Return_Temp": chw_return_temp,
        "CW_Supply_Temp": cw_supply_temp,
        "CW_Return_Temp": cw_return_temp,
        "CHWP_Hz": chwp_freq,
        "CWP_Hz": cwp_freq,
        "CT_Hz": ct_freq,
        "CH_kW": chiller_kw,
        "CHWP_kW": chwp_kw,
        "CWP_kW": cwp_kw,
        "CT_kW": ct_kw,
        "Total_kW": total_power
    })
    
    # Issue 1: Spikes
    spike_indices = np.random.choice(n, 10, replace=False)
    df.loc[spike_indices, "Total_kW"] *= 5.0
    df.loc[spike_indices[:3], "CHW_Supply_Temp"] = 99.0
    
    # Issue 2: Flatlines (Sensor stuck)
    # Day 5, 10:00 to 14:00, OA_Temp stays same
    flat_mask = (timestamps >= datetime(2024, 7, 5, 10, 0)) & (timestamps <= datetime(2024, 7, 5, 14, 0))
    df.loc[flat_mask, "OA_Temp"] = df.loc[flat_mask, "OA_Temp"].iloc[0]
    
    # Issue 3: Range Violations
    df.loc[100, "OA_RH"] = -10.0
    df.loc[101, "CWP_Hz"] = 500.0
    
    # Issue 4: Shutdowns (Simulate night shutdown for 1st week)
    night_mask = (timestamps.hour >= 23) | (timestamps.hour <= 5)
    first_week_mask = (timestamps.day <= 7)
    df.loc[night_mask & first_week_mask, "Total_kW"] = 2.0 # Only monitoring
    df.loc[night_mask & first_week_mask, ["CH_kW", "CHWP_kW", "CWP_kW", "CT_kW"]] = 0.0
    
    # Issue 5: Energy Balance Mismatch
    # Day 15, all components reported but Total_kW is artificially low
    mismatch_mask = (timestamps >= datetime(2024, 7, 15, 12, 0)) & (timestamps <= datetime(2024, 7, 15, 13, 0))
    df.loc[mismatch_mask, "Total_kW"] = 100.0 # Way too low
    
    # Issue 6: Missing Gaps (Data loss)
    gap_mask = (timestamps >= datetime(2024, 7, 10, 8, 0)) & (timestamps <= datetime(2024, 7, 10, 8, 30)) # 30 min gap
    df = df.drop(df[gap_mask].index).reset_index(drop=True)
    
    # Save to CSV
    output_path = "data/hvac_monthly_real.csv"
    df.to_csv(output_path, index=False)
    print(f"Dataset generated at {output_path}. Total rows: {len(df)}")

if __name__ == "__main__":
    generate_hvac_dataset()
