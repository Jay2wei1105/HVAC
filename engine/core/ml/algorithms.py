# engine/core/ml/algorithms.py
from typing import Callable
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def _xgb_space(trial):
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 200, 1500, step=100),
        max_depth=trial.suggest_int("max_depth", 4, 10),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        random_state=42, tree_method="hist")

def _lgbm_space(trial):
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 200, 1500, step=100),
        num_leaves=trial.suggest_int("num_leaves", 15, 127),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        feature_fraction=trial.suggest_float("feature_fraction", 0.6, 1.0),
        bagging_fraction=trial.suggest_float("bagging_fraction", 0.6, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        random_state=42, verbose=-1)

def _rf_space(trial):
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 800, step=50),
        max_depth=trial.suggest_int("max_depth", 5, 25),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
        max_features=trial.suggest_float("max_features", 0.3, 1.0),
        random_state=42, n_jobs=-1)

def _mlp_space(trial):
    h1 = trial.suggest_int("h1", 32, 256, step=32)
    h2 = trial.suggest_int("h2", 0, 128, step=32)  # 0 = 單層
    layers = (h1,) if h2 == 0 else (h1, h2)
    return dict(
        hidden_layer_sizes=layers,
        alpha=trial.suggest_float("alpha", 1e-5, 1e-2, log=True),
        learning_rate_init=trial.suggest_float("lr_init", 1e-4, 1e-2, log=True),
        max_iter=500, early_stopping=True, random_state=42)

# (Estimator 工廠, Optuna 搜索空間, 是否需 scaling)
ALGORITHMS: dict[str, tuple[Callable, Callable, bool]] = {
    "xgboost":       (XGBRegressor,          _xgb_space,  False),
    "lightgbm":      (LGBMRegressor,         _lgbm_space, False),
    "random_forest": (RandomForestRegressor, _rf_space,   False),
    "mlp":           (MLPRegressor,          _mlp_space,  True),   # ANN 必須 scaling
}

def make_estimator(algo: str, params: dict):
    cls, _, need_scale = ALGORITHMS[algo]
    est = cls(**params)
    if need_scale:
        return Pipeline([("scaler", StandardScaler()), ("model", est)])
    return est
