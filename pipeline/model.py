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


WEIGHT_MODES = ("none", "abs", "sqrt")


def sample_weights(returns, mode: str = "none"):
    """
    Gewichte je Trainingszeile aus der realisierten Rendite.

    Das Label `win` bewertet eine Bewegung von 0.001 genauso wie eine von 0.05,
    waehrend der Gewinn ausschliesslich im Schwanz steckt: der mittlere Trade
    verliert 0.16 %, und ohne die besten 50 von 7.585 Trades kippt das Ergebnis
    ins Minus.  Gewichte proportional zu |trade_return| richten den
    Klassifikator auf die Bewegungen aus, an denen tatsaechlich verdient wird —
    ohne Ziel, Auswertung, Backtest oder Live-Pfad anzufassen.

    `sqrt` ist die vorsichtigere Variante: sie hebt grosse Bewegungen hervor,
    ohne das Training von einzelnen Ausreissern beherrschen zu lassen.

    Zwei Details, die den Vergleich ueberhaupt erst zulassen:

    * **Normiert auf Mittelwert 1.**  LightGBM skaliert Gradienten mit dem
      Gewicht, also wirkte ein ungewichteter Lauf sonst wie ein anderer
      Lernraten- und Regularisierungspunkt.  Ohne die Normierung vergliche ein
      Gitter Gewichtung und Lernrate durcheinander.
    * **Untergrenze 0.01.**  Ein Gewicht von exakt 0 entfernt die Zeile
      vollstaendig aus dem Training.  Zeilen in der Totzone haben |return| nahe
      0 und wuerden genau so verschwinden — die Gewichtung soll betonen, nicht
      wegwerfen.
    """
    if mode in (None, "none"):
        return None
    if mode not in WEIGHT_MODES:
        raise ValueError(f"unbekannter Gewichtungsmodus {mode!r}, erlaubt: {WEIGHT_MODES}")
    if returns is None:
        raise ValueError(f"Gewichtung {mode!r} verlangt trade_return, bekam None")

    w = np.abs(np.nan_to_num(np.asarray(returns, dtype=np.float64), nan=0.0,
                             posinf=0.0, neginf=0.0))
    if mode == "sqrt":
        w = np.sqrt(w)
    mean = w.mean()
    if not np.isfinite(mean) or mean <= 0:
        return None
    return np.clip(w / mean, 0.01, None)


def train_model(
    split: SplitResult,
    model_cfg: ModelConfig,
    monitor_cfg: MonitorConfig,
    save_path: Optional[str] = "model.txt",
    weight_mode: str = "none",
) -> tuple[lgb.Booster, TrainingMonitor]:
    """
    Train a LightGBM model with live monitoring.

    Returns the trained booster and the monitor (for history/summary).
    """

    # Create datasets
    #
    # Die Validierung wird bewusst MIT denselben Gewichten versehen: das Early
    # Stopping soll nach derselben Groesse entscheiden, auf die trainiert wird.
    # Ungewichtet zu validieren hiesse, auf ein Ziel zu optimieren und nach
    # einem anderen abzubrechen.
    train_w = sample_weights(split.train_ret, weight_mode)
    val_w = sample_weights(split.val_ret, weight_mode)

    train_data = lgb.Dataset(
        split.train_X,
        label=split.train_y,
        weight=train_w,
        feature_name=split.feature_names,
        free_raw_data=False,
    )

    val_data = lgb.Dataset(
        split.val_X,
        label=split.val_y,
        weight=val_w,
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
