# engine/core/ml/trainer.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

# Patch XGBRegressor to bypass strict scikit-learn/mlflow autolog checks
XGBRegressor._estimator_type = "regressor"


@dataclass
class TrainResult:
    model: XGBRegressor
    best_params: dict[str, Any]
    cv_mape_best: float
    holdout_mape: float
    holdout_cv_rmse: float
    holdout_mae: float
    feature_importance: dict[str, float] = field(default_factory=dict)
    shap_vals: np.ndarray | None = None
    shap_expected_value: float = 0.0
    shap_sample: pd.DataFrame | None = None
    mlflow_run_id: str = ""


def _try_import_mlflow() -> Any | None:
    """Return mlflow module if installed; otherwise None (training still runs)."""
    try:
        import mlflow

        return mlflow
    except ImportError:
        # ModuleNotFoundError is a subclass; some envs raise ImportError for meta-path hooks
        return None


def _try_import_shap() -> Any | None:
    """Return shap module if installed; otherwise None."""
    try:
        import shap

        return shap
    except ImportError:
        return None


def _require_optuna() -> tuple[Any, Any, Any]:
    """Lazy import so importing this module does not require Optuna installed."""
    try:
        import optuna
        from optuna.pruners import MedianPruner
        from optuna.samplers import TPESampler
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Optuna is required for training. Install with: pip install optuna"
        ) from exc
    return optuna, MedianPruner, TPESampler


def _dbg(msg: str, data: dict) -> None:  # noqa: D401
    # Best-effort debug logging only; never block training on filesystem issues.
    try:
        import json, time as _time, pathlib

        _p = pathlib.Path("/Users/jay/Downloads/missionnnn/.cursor/debug-67042e.log")
        _p.parent.mkdir(parents=True, exist_ok=True)
        entry = {"sessionId": "67042e", "timestamp": int(_time.time() * 1000),
                 "location": "trainer.py", "message": msg, "data": data}
        with _p.open("a") as _f:
            _f.write(json.dumps(entry) + "\n")
    except Exception:
        return


def train_and_evaluate(
    df: pd.DataFrame,
    addon: Any,
    target: str = "total_power",
    n_trials: int = 100,
    cv_folds: int = 5,
    gap: int = 4,
    holdout_ratio: float = 0.2,
    experiment_tag: str = "baseline",
) -> TrainResult:
    """Run Optuna HPO + final XGBoost fit; MLflow / SHAP are optional."""
    mlflow = _try_import_mlflow()
    shap = _try_import_shap()
    optuna, MedianPruner, TPESampler = _require_optuna()

    feat_df = addon.prediction.build_features(df, target=target)
    feat_cols = addon.prediction.get_feature_columns(feat_df, target=target)
    X, y = feat_df[feat_cols], feat_df[target]

    # #region agent log — H-A/H-E: inspect target dtype before split
    _dbg("target_col_info", {
        "hypothesisId": "H-A,H-E",
        "target": target,
        "dtype": str(y.dtype),
        "n_rows": len(y),
        "sample_head": [str(v) for v in y.head(3).tolist()],
        "sample_tail": [str(v) for v in y.tail(3).tolist()],
        "has_nulls": int(y.isna().sum()),
    })
    # #endregion

    # 1. Holdout split（時序不打亂：最後 20% 是 holdout）
    split = int(len(X) * (1 - holdout_ratio))
    X_tr, X_ho = X.iloc[:split], X.iloc[split:]
    y_tr, y_ho = y.iloc[:split], y.iloc[split:]

    if mlflow is not None:
        mlflow.end_run()
        mlflow.sklearn.autolog(disable=True)

    def objective(
        trial: Any,
        X_tr_inner: pd.DataFrame,
        y_tr_inner: pd.Series,
        cv_folds_inner: int,
        gap_inner: int,
    ) -> float:
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 1500, step=100),
            max_depth=trial.suggest_int("max_depth", 4, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            random_state=42,
            tree_method="hist",
            early_stopping_rounds=50,
        )
        tscv = TimeSeriesSplit(n_splits=cv_folds_inner, gap=gap_inner)
        mapes: list[float] = []
        for fold, (tr, va) in enumerate(tscv.split(X_tr_inner)):
            m = XGBRegressor(**params)
            m.fit(
                X_tr_inner.iloc[tr],
                y_tr_inner.iloc[tr],
                eval_set=[(X_tr_inner.iloc[va], y_tr_inner.iloc[va])],
                verbose=False,
            )
            mapes.append(
                mean_absolute_percentage_error(
                    y_tr_inner.iloc[va], m.predict(X_tr_inner.iloc[va])
                )
            )
            trial.report(float(np.mean(mapes)), step=fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(mapes))

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_warmup_steps=2, n_startup_trials=10),
    )
    callbacks: list[Any] = []
    if mlflow is not None:
        try:
            from optuna_integration.mlflow import MLflowCallback as _MLflowCB
        except ImportError:
            try:
                from optuna.integration.mlflow import MLflowCallback as _MLflowCB  # type: ignore[no-redef]
            except ImportError:
                _MLflowCB = None  # type: ignore[assignment,misc]
        if _MLflowCB is not None:
            callbacks.append(
                _MLflowCB(
                    tracking_uri=mlflow.get_tracking_uri(),
                    metric_name="cv_mape",
                )
            )

    study.optimize(
        lambda t: objective(t, X_tr, y_tr, cv_folds, gap),
        n_trials=n_trials,
        callbacks=callbacks,
        show_progress_bar=True,
    )

    best = {**study.best_params, "random_state": 42, "tree_method": "hist"}
    final = XGBRegressor(**best)
    final.fit(X_tr, y_tr, verbose=False)

    y_hat = final.predict(X_ho)

    # #region agent log — H-A/H-B: inspect y_ho and y_hat before metric calc
    _dbg("pre_metric_check", {
        "hypothesisId": "H-A,H-B",
        "y_ho_dtype": str(y_ho.dtype),
        "y_ho_sample": [str(v) for v in y_ho.iloc[:3].tolist()],
        "y_hat_type": str(type(y_hat).__name__),
        "y_hat_dtype": str(y_hat.dtype),
        "y_hat_sample": [str(v) for v in y_hat[:3].tolist()],
        "y_ho_mean": str(y_ho.mean()),
    })
    # #endregion

    ho_mape = mean_absolute_percentage_error(y_ho, y_hat)
    ho_rmse = float(np.sqrt(mean_squared_error(y_ho, y_hat)))
    ho_mae = mean_absolute_error(y_ho, y_hat)
    ho_cvrmse = ho_rmse / float(y_ho.mean())

    # #region agent log — metrics computed successfully
    _dbg("metrics_ok", {
        "hypothesisId": "H-A,H-B",
        "ho_mape": float(ho_mape), "ho_rmse": ho_rmse,
        "ho_mae": float(ho_mae), "ho_cvrmse": float(ho_cvrmse),
    })
    # #endregion

    sample = X_ho.sample(min(500, len(X_ho)), random_state=42)
    shap_vals: np.ndarray | None = None
    exp_val = 0.0
    if shap is not None:
        explainer = shap.TreeExplainer(final)
        shap_vals = explainer.shap_values(sample)
        # #region agent log — H-C: shap expected_value type
        _dbg("shap_expected_value", {
            "hypothesisId": "H-C",
            "expected_value_type": str(type(explainer.expected_value).__name__),
            "expected_value_repr": str(explainer.expected_value)[:100],
        })
        # #endregion
        if isinstance(explainer.expected_value, np.ndarray):
            exp_val = float(explainer.expected_value[0])
        else:
            exp_val = float(explainer.expected_value)

    run_id = ""
    if mlflow is not None:
        with mlflow.start_run(run_name=f"{experiment_tag}_final") as run:
            mlflow.log_metrics(
                {
                    "cv_mape_best": float(study.best_value),
                    "holdout_mape": float(ho_mape),
                    "holdout_cv_rmse": float(ho_cvrmse),
                    "holdout_mae": float(ho_mae),
                }
            )
            mlflow.log_params({k: float(v) if isinstance(v, (int, float)) else v
                               for k, v in best.items()})
            if shap_vals is not None:
                top = dict(
                    sorted(
                        zip(feat_cols, np.abs(shap_vals).mean(0).tolist()),
                        key=lambda kv: -kv[1],
                    )[:20]
                )
                mlflow.log_dict(top, "shap_top20.json")
            try:
                mlflow.xgboost.log_model(
                    final,
                    "model",
                    registered_model_name=f"{addon.__class__.__name__}-{target}",
                )
            except Exception:
                pass  # model logging is best-effort; don't abort training
            run_id = run.info.run_id
        mlflow.end_run()

    return TrainResult(
        model=final,
        best_params=study.best_params,
        cv_mape_best=study.best_value,
        holdout_mape=ho_mape,
        holdout_cv_rmse=ho_cvrmse,
        holdout_mae=ho_mae,
        feature_importance=dict(zip(feat_cols, final.feature_importances_.tolist())),
        shap_vals=shap_vals,
        shap_expected_value=exp_val,
        shap_sample=sample if shap_vals is not None else None,
        mlflow_run_id=run_id,
    )
