"""MASE / RMSSE on a forecast panel. Scale is seasonal naive on the train tail."""

from __future__ import annotations

import numpy as np
import pandas as pd

SEASON = 52


def scaled_errors(
    actual: pd.DataFrame,
    pred: pd.DataFrame,
    train: pd.DataFrame,
    model_col: str,
    season: int = SEASON,
) -> pd.DataFrame:
    """One row per unique_id with MASE and RMSSE for `model_col`."""
    rows = []
    yhat = pred.set_index(["unique_id", "ds"])[model_col]
    y = actual.set_index(["unique_id", "ds"])["y"]
    joined = pd.concat([y.rename("y"), yhat.rename("yhat")], axis=1).dropna()
    for uid, grp in joined.groupby(level=0):
        hist = train.loc[train["unique_id"].eq(uid), "y"].to_numpy(dtype=float)
        scale = _seasonal_scale(hist, season)
        err = grp["y"].to_numpy() - grp["yhat"].to_numpy()
        rows.append(
            {
                "unique_id": uid,
                "mase": float(np.mean(np.abs(err)) / scale),
                "rmsse": float(np.sqrt(np.mean(err**2)) / scale),
                "n": int(len(err)),
            }
        )
    return pd.DataFrame(rows)


def _seasonal_scale(y: np.ndarray, season: int) -> float:
    if len(y) > season:
        scale = np.mean(np.abs(y[season:] - y[:-season]))
    else:
        scale = np.mean(np.abs(np.diff(y))) if len(y) > 1 else 0.0
    return float(scale) if scale > 0 else 1.0
