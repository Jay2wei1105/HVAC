from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


@dataclass
class QDemandTrainResult:
    model: Any
    best_params: dict[str, Any]
    pinball_holdout: float
    baseline_pinball_holdout: float
    pinball_improvement: float
    coverage: float
    feature_importance: dict[str, float] = field(default_factory=dict)
    top_features: list[str] = field(default_factory=list)
    validation: dict[str, bool] = field(default_factory=dict)


def _require_lightgbm() -> Any:
    try:
        from lightgbm import LGBMRegressor
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "LightGBM is required for q_demand training. Install with: pip install lightgbm"
        ) from exc
    return LGBMRegressor


def _require_optuna() -> tuple[Any, Any, Any]:
    try:
        import optuna
        from optuna.pruners import MedianPruner
        from optuna.samplers import TPESampler
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Optuna is required for q_demand training. Install with: pip install optuna"
        ) from exc
    return optuna, MedianPruner, TPESampler


def pinball_loss(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    alpha: float = 0.9,
) -> float:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    diff = y_true_arr - y_pred_arr
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1) * diff)))


def _try_import_shap() -> Any | None:
    try:
        import shap

        return shap
    except ImportError:
        return None


def _baseline_recent_mean_prediction(
    y_all: pd.Series,
    holdout_start: int,
    horizon_len: int,
    lookback_rows: int = 96,
) -> np.ndarray:
    preds: list[float] = []
    y_all = y_all.astype(float)
    for offset in range(horizon_len):
        pos = holdout_start + offset
        window = y_all.iloc[max(0, pos - lookback_rows):pos]
        preds.append(float(window.mean()))
    return np.asarray(preds, dtype=float)


def train_q_demand_model(
    df: pd.DataFrame,
    addon: Any,
    n_trials: int = 80,
    alpha: float = 0.9,
    gap: int = 4,
    cv_folds: int = 5,
    holdout_ratio: float = 0.2,
) -> QDemandTrainResult:
    LGBMRegressor = _require_lightgbm()
    optuna, MedianPruner, TPESampler = _require_optuna()
    shap = _try_import_shap()

    feat_df = addon.q_demand.build_features(df)
    feat_cols = addon.q_demand.get_feature_columns(feat_df)
    X, y = feat_df[feat_cols], feat_df["q_delivered_kw"].astype(float)
    min_required = max(12, gap + 6)
    if len(X) < min_required:
        raise ValueError(
            f"Not enough q_demand samples after feature building ({len(X)} rows). Need at least {min_required}."
        )

    effective_cv_folds = min(cv_folds, max(2, len(X) // 8))
    effective_gap = min(gap, max(0, len(X) // 20))

    split = int(len(X) * (1 - holdout_ratio))
    if split <= effective_cv_folds + effective_gap + 1:
        split = len(X) - max(4, len(X) // 5)
    if split <= effective_cv_folds + effective_gap + 1 or split >= len(X):
        raise ValueError(
            f"Invalid q_demand holdout split after feature building ({len(X)} rows)."
        )
    X_tr, X_ho = X.iloc[:split], X.iloc[split:]
    y_tr, y_ho = y.iloc[:split], y.iloc[split:]

    def objective(trial: Any) -> float:
        params = {
            "objective": "quantile",
            "alpha": alpha,
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=100),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "random_state": 42,
            "verbose": -1,
        }
        tscv = TimeSeriesSplit(n_splits=effective_cv_folds, gap=effective_gap)
        losses: list[float] = []
        for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_tr)):
            model = LGBMRegressor(**params)
            model.fit(X_tr.iloc[tr_idx], y_tr.iloc[tr_idx])
            pred = model.predict(X_tr.iloc[va_idx])
            losses.append(pinball_loss(y_tr.iloc[va_idx], pred, alpha))
            trial.report(float(np.mean(losses)), step=fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(losses))

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_warmup_steps=2, n_startup_trials=10),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    final_params = {
        **study.best_params,
        "objective": "quantile",
        "alpha": alpha,
        "random_state": 42,
        "verbose": -1,
    }
    final = LGBMRegressor(**final_params)
    final.fit(X_tr, y_tr)

    y_hat = final.predict(X_ho)
    coverage = float((y_ho <= y_hat).mean())
    holdout_pinball = pinball_loss(y_ho, y_hat, alpha)

    baseline_pred = _baseline_recent_mean_prediction(y, split, len(X_ho))
    baseline_pinball = pinball_loss(y_ho, baseline_pred, alpha)
    improvement = 0.0 if baseline_pinball == 0 else (baseline_pinball - holdout_pinball) / baseline_pinball
    feature_importance = dict(zip(feat_cols, final.feature_importances_.tolist()))
    top_features = [
        name
        for name, _ in sorted(feature_importance.items(), key=lambda item: -item[1])[:3]
    ]

    if shap is not None and len(X_ho) > 0:
        sample = X_ho.sample(min(200, len(X_ho)), random_state=42)
        explainer = shap.TreeExplainer(final)
        shap_vals = explainer.shap_values(sample)
        top_features = [
            name
            for name, _ in sorted(
                zip(feat_cols, np.abs(shap_vals).mean(0).tolist()),
                key=lambda item: -item[1],
            )[:3]
        ]

    validation = {
        "coverage_ok": 0.85 <= coverage <= 0.93,
        "pinball_improvement_ok": improvement >= 0.30,
        "weather_driver_ok": any(
            feature in top_features for feature in ("outdoor_temp", "wet_bulb")
        ),
    }

    return QDemandTrainResult(
        model=final,
        best_params=study.best_params,
        pinball_holdout=holdout_pinball,
        baseline_pinball_holdout=baseline_pinball,
        pinball_improvement=float(improvement),
        coverage=coverage,
        feature_importance=feature_importance,
        top_features=top_features,
        validation=validation,
    )
