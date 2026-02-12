"""
LightGBM model training with live monitoring integration.
"""

import numpy as np
import lightgbm as lgb
from pathlib import Path
from typing import Optional

from config import ModelConfig, MonitorConfig
from pipeline.monitor import TrainingMonitor
from pipeline.splitter import SplitResult


def train_model(
    split: SplitResult,
    model_cfg: ModelConfig,
    monitor_cfg: MonitorConfig,
    save_path: Optional[str] = "model.txt",
) -> tuple[lgb.Booster, TrainingMonitor]:
    """
    Train a LightGBM model with live monitoring.

    Returns the trained booster and the monitor (for history/summary).
    """

    # Create datasets
    train_data = lgb.Dataset(
        split.train_X,
        label=split.train_y,
        feature_name=split.feature_names,
        free_raw_data=False,
    )

    val_data = lgb.Dataset(
        split.val_X,
        label=split.val_y,
        feature_name=split.feature_names,
        reference=train_data,
        free_raw_data=False,
    )

    # Setup monitor
    monitor = TrainingMonitor(monitor_cfg)
    monitor.set_feature_names(split.feature_names)
    monitor.set_total_iterations(model_cfg.n_estimators)

    # LightGBM parameters
    params = model_cfg.to_lgbm_params()

    # Remove sklearn-style params that lgb.train doesn't accept
    n_estimators = params.pop("n_estimators", 5000)
    params.pop("verbose", None)

    # Add metrics
    params["metric"] = ["binary_logloss", "auc"]
    params["verbose"] = -1
    params["seed"] = 42

    # Train
    callbacks = [
        monitor.callback(),
        lgb.early_stopping(model_cfg.early_stopping_rounds, verbose=False),
        lgb.log_evaluation(period=0),  # Suppress default logging
    ]

    booster = lgb.train(
        params,
        train_data,
        num_boost_round=n_estimators,
        valid_sets=[train_data, val_data],
        valid_names=["train", "valid_0"],
        callbacks=callbacks,
    )

    # Finish monitoring
    monitor.finish()

    # Save model
    if save_path:
        booster.save_model(save_path)
        print(f"Model saved to {save_path}")

    return booster, monitor


def predict(
    booster: lgb.Booster,
    X: np.ndarray,
) -> np.ndarray:
    """Predict P(win) probabilities."""
    return booster.predict(X)


def feature_importance(
    booster: lgb.Booster,
    feature_names: list[str],
    importance_type: str = "gain",
    top_n: int = 20,
) -> list[tuple[str, float]]:
    """Return sorted feature importances."""

    importance = booster.feature_importance(importance_type=importance_type)
    pairs = sorted(
        zip(feature_names, importance),
        key=lambda x: x[1],
        reverse=True,
    )
    return pairs[:top_n]


def load_model(path: str) -> lgb.Booster:
    """Load a saved model."""
    return lgb.Booster(model_file=path)
