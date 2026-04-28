import pandas as pd
import numpy as np
import os
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

class MLService:
    @staticmethod
    def _calculate_total_power(df: pd.DataFrame, mappings: list) -> pd.Series:
        # Check if total_power is explicitly mapped
        for m in mappings:
            if m["target"] == "total_power":
                return df[m["source"]]
        
        # Fallback: Sum all mapped power columns
        power_sources = [m["source"] for m in mappings if m["target"] and "power" in m["target"]]
        if not power_sources:
            # Absolute fallback: find any column with 'kw' or 'power'
            power_sources = [c for c in df.columns if 'kw' in c.lower() or 'power' in c.lower() or '電力' in c]
            
        if not power_sources:
            return pd.Series(np.zeros(len(df)), index=df.index)
            
        # Hard Sum
        return df[power_sources].sum(axis=1)

    @staticmethod
    def _construct_features(df: pd.DataFrame, mappings: list) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        
        # Identify mapped columns
        mapped_dict = {m["target"]: m["source"] for m in mappings if m["target"]}
        
        # 1. Feature: Time-based
        if "timestamp" in mapped_dict:
            ts_col = df[mapped_dict["timestamp"]]
            # Ensure datetime type
            ts = pd.to_datetime(ts_col, errors='coerce')
            features['hour'] = ts.dt.hour
            features['is_weekend'] = ts.dt.dayofweek >= 5
            features['month'] = ts.dt.month
        
        # 2. Extract base features
        base_features = []
        if "ambient_temp" in mapped_dict:
            features['oa_t'] = df[mapped_dict["ambient_temp"]]
            base_features.append('oa_t')
        
        if "oa_rh" in mapped_dict:
            features['oa_rh'] = df[mapped_dict["oa_rh"]]
            base_features.append('oa_rh')
        else:
            # Fallback if RH is missing, assume constant 70%
            features['oa_rh'] = 70.0
            
        if "chws_temp" in mapped_dict:
            features['chws'] = df[mapped_dict["chws_temp"]]
            
        if "flow_rate" in mapped_dict:
            features['flow'] = df[mapped_dict["flow_rate"]]
            
        # 3. Time-lag & Rolling Effects for Thermal Mass (if time context exists)
        # Assuming data is sorted and roughly continuous
        for bf in base_features:
            # Fillna just in case
            features[bf] = features[bf].ffill().bfill()
            
            # 1-hour lag (assuming 1 row = 1 min or 1 hour; we'll do raw row shifts pretending hourly. 
            # If 15-min intervals, shift(4). To be robust, let's just do shift(1), shift(3) etc)
            # A more robust way is just shift by rows.
            features[f'{bf}_lag_1'] = features[bf].shift(1).bfill()
            features[f'{bf}_lag_3'] = features[bf].shift(3).bfill()
            features[f'{bf}_roll_mean_6'] = features[bf].rolling(window=6, min_periods=1).mean()
            features[f'{bf}_delta'] = features[bf] - features[f'{bf}_lag_1']

        # Drop any remaining NaNs
        features = features.fillna(0)
        return features

    @staticmethod
    def train_baseline_model(site_id: str, df: pd.DataFrame, mappings: list, storage_dir: str):
        try:
            # 1. Target Y
            y = MLService._calculate_total_power(df, mappings)
            
            # 2. Features X
            X = MLService._construct_features(df, mappings)
            
            # If X is empty or y is all zeros, return dummy
            if X.empty or len(X.columns) == 0 or y.sum() == 0:
                raise ValueError("Insufficient features or target power to train model.")
            
            # Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
            
            # Train XGBoost
            model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate
            predictions = model.predict(X_test)
            r2 = r2_score(y_test, predictions)
            
            # Feature Importance
            importance = model.feature_importances_
            feature_names = X.columns
            # Sort by importance
            feat_imp = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)
            top_features = [{"feature": f[0], "importance": float(f[1])} for f in feat_imp[:5]]
            
            # Save Model
            model_path = os.path.join(storage_dir, f"baseline_model_{site_id}.joblib")
            joblib.dump(model, model_path)
            
            return {
                "status": "success",
                "r2_score": float(r2),
                "top_features": top_features,
                "model_path": model_path
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
