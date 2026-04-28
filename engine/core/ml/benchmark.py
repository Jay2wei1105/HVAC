# engine/core/ml/benchmark.py
import pandas as pd
import optuna
import mlflow
from typing import List
from .algorithms import ALGORITHMS, make_estimator
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error
import numpy as np

def run_leaderboard(
    df: pd.DataFrame, addon, target: str,
    algos: List[str] = ["xgboost", "lightgbm", "random_forest"],
    trials_per_algo: int = 20,
) -> pd.DataFrame:
    """自動跑多個演算法，產出排行榜供決策。"""
    res = []
    feat_df = addon.prediction.build_features(df, target)
    feat_cols = addon.prediction.get_feature_columns(feat_df, target)
    X, y = feat_df[feat_cols], feat_df[target]
    
    # 簡化版訓練（不分 holdout，快速跑量）
    for algo in algos:
        if algo not in ALGORITHMS: continue
        _, space_fn, _ = ALGORITHMS[algo]
        
        def objective(trial):
            params = space_fn(trial)
            tscv = TimeSeriesSplit(n_splits=3)
            mapes = []
            for tr, va in tscv.split(X):
                m = make_estimator(algo, params)
                # XGB/LGBM 需要特別處理 eval_set，這裡簡化只用 generic fit
                m.fit(X.iloc[tr], y.iloc[tr])
                mapes.append(mean_absolute_percentage_error(y.iloc[va], m.predict(X.iloc[va])))
            return float(np.mean(mapes))

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=trials_per_algo)
        
        res.append({
            "algorithm": algo,
            "best_mape": study.best_value,
            "best_params": study.best_params
        })
        
    lb = pd.DataFrame(res).sort_values("best_mape")
    return lb
