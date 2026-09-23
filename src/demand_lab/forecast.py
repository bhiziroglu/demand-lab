"""Global LightGBM (MLForecast) vs SeasonalNaive, plus SHAP-style drivers and a promo what-if."""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive

from demand_lab.evaluate import SEASON, scaled_errors

def _weekofyear(dates) -> pd.Series:
    idx = dates if isinstance(dates, pd.DatetimeIndex) else pd.DatetimeIndex(dates)
    return idx.isocalendar().week.astype("int16")


HORIZON = 8
N_WINDOWS = 4
LGBM_NAME = "LGBMRegressor"
NAIVE_NAME = "SeasonalNaive"
STATIC = ["item_code", "store_code", "list_price"]
EXOG = ["discount", "on_promo", "snap", "event"]

FEATURE_LABELS = {
    "rolling_mean_lag1_window_size4": "avg units, last 4 weeks",
    "rolling_mean_lag1_window_size13": "avg units, last 13 weeks",
    "lag1": "units last week",
    "lag2": "units 2 weeks ago",
    "lag4": "units 4 weeks ago",
    "lag13": "units 13 weeks ago",
    "item_code": "item",
    "store_code": "store",
    "list_price": "regular price",
    "discount": "discount vs regular",
    "on_promo": "on promo",
    "snap": "SNAP week",
    "event": "event week",
    "month": "month",
    "_weekofyear": "week of year",
}


@dataclass
class Backtest:
    panel: pd.DataFrame
    lgbm_cv: pd.DataFrame
    naive_cv: pd.DataFrame
    metrics: pd.DataFrame
    forecast: MLForecast
    drivers: pd.DataFrame
    whatif: pd.DataFrame
    sample_id: str
    pi: dict


def run_backtest(panel: pd.DataFrame) -> Backtest:
    panel = panel.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    fcst = _lgbm()
    lgbm_cv = fcst.cross_validation(
        panel[["unique_id", "ds", "y", *STATIC, *EXOG]],
        n_windows=N_WINDOWS,
        h=HORIZON,
        step_size=HORIZON,
        static_features=STATIC,
    )
    naive = StatsForecast(
        models=[SeasonalNaive(season_length=SEASON)],
        freq="W-SAT",
        n_jobs=1,
    )
    naive_cv = naive.cross_validation(
        h=HORIZON,
        df=panel[["unique_id", "ds", "y"]],
        step_size=HORIZON,
        n_windows=N_WINDOWS,
    ).dropna(subset=[NAIVE_NAME])
    last_cutoff = lgbm_cv["cutoff"].max()
    last_lgbm = lgbm_cv[lgbm_cv["cutoff"].eq(last_cutoff)].copy()
    last_naive = naive_cv[naive_cv["cutoff"].eq(last_cutoff)].copy()
    train_scale = panel[panel["ds"] <= lgbm_cv["cutoff"].min()]
    metrics = _metrics_table(panel, lgbm_cv, naive_cv, train_scale)
    fcst.fit(panel[["unique_id", "ds", "y", *STATIC, *EXOG]], static_features=STATIC)
    sample_id = _pick_sample(panel, last_lgbm)
    pi = _prediction_intervals(lgbm_cv, panel, sample_id)
    drivers = _driver_table(fcst, panel, sample_id)
    whatif = _promo_whatif(fcst, panel, sample_id)
    return Backtest(
        panel=panel,
        lgbm_cv=lgbm_cv,
        naive_cv=naive_cv,
        metrics=metrics,
        forecast=fcst,
        drivers=drivers,
        whatif=whatif,
        sample_id=sample_id,
        pi=pi,
    )


def _lgbm() -> MLForecast:
    model = lgb.LGBMRegressor(
        n_estimators=200,
        num_leaves=31,
        learning_rate=0.05,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=0,
        n_jobs=-1,
        verbosity=-1,
    )
    return MLForecast(
        models=model,
        freq="W-SAT",
        lags=[1, 2, 4, 13],
        lag_transforms={1: [RollingMean(window_size=4), RollingMean(window_size=13)]},
        date_features=["month", _weekofyear],
        num_threads=1,
    )


def _metrics_table(
    panel: pd.DataFrame,
    lgbm_cv: pd.DataFrame,
    naive_cv: pd.DataFrame,
    train: pd.DataFrame,
) -> pd.DataFrame:
    actual = lgbm_cv[["unique_id", "ds", "y"]]
    rows = []
    for name, pred, col in (
        ("LightGBM", lgbm_cv, LGBM_NAME),
        ("SeasonalNaive", naive_cv, NAIVE_NAME),
    ):
        scores = scaled_errors(actual, pred, train, col)
        rows.append(_summary(name, "all", scores, pred, panel))
        promo = pred.merge(panel[["unique_id", "ds", "on_promo"]], on=["unique_id", "ds"])
        for flag, label in ((1, "promo"), (0, "no_promo")):
            part = promo[promo["on_promo"].eq(flag)]
            if part.empty:
                continue
            part_scores = scaled_errors(part[["unique_id", "ds", "y"]], part, train, col)
            rows.append(_summary(name, label, part_scores, part, panel))
    return pd.DataFrame(rows)


def _summary(
    model: str,
    bucket: str,
    scores: pd.DataFrame,
    pred: pd.DataFrame,
    panel: pd.DataFrame,
) -> dict:
    return {
        "model": model,
        "bucket": bucket,
        "mase": round(float(scores["mase"].mean()), 3),
        "rmsse": round(float(scores["rmsse"].mean()), 3),
        "series": int(scores["unique_id"].nunique()),
        "rows": int(len(pred)),
        "promo_share": round(float(panel["on_promo"].mean()), 3),
    }


def residual_quantiles(resid: pd.Series, lo: float = 0.1, hi: float = 0.9) -> tuple[float, float]:
    """10th/90th residual percentiles for an 80% band around a point forecast."""
    clean = resid.dropna()
    if clean.empty:
        return 0.0, 0.0
    return float(clean.quantile(lo)), float(clean.quantile(hi))


def _prediction_intervals(
    lgbm_cv: pd.DataFrame, panel: pd.DataFrame, sample_id: str
) -> dict:
    # ponytail: empirical CV residuals, not a density. Ignores residual autocorrelation;
    # upgrade is split conformal or LightGBM quantile heads.
    cv = lgbm_cv.copy()
    cv["resid"] = cv["y"] - cv[LGBM_NAME]
    sku_lo, sku_hi = residual_quantiles(cv.loc[cv["unique_id"].eq(sample_id), "resid"])
    store = panel.loc[panel["unique_id"].eq(sample_id), "store_id"].iloc[0]
    rolled = (
        cv.merge(panel[["unique_id", "ds", "store_id"]], on=["unique_id", "ds"])
        .loc[lambda d: d["store_id"].eq(store)]
        .groupby(["cutoff", "ds"], as_index=False)
        .agg(y=("y", "sum"), yhat=(LGBM_NAME, "sum"))
    )
    store_lo, store_hi = residual_quantiles(rolled["y"] - rolled["yhat"])
    return {
        "sku_lo": sku_lo,
        "sku_hi": sku_hi,
        "store_lo": store_lo,
        "store_hi": store_hi,
    }


def _pick_sample(panel: pd.DataFrame, last_lgbm: pd.DataFrame) -> str:
    vol = panel.groupby("unique_id", observed=True)["y"].sum().sort_values(ascending=False)
    err = last_lgbm.copy()
    err["abs"] = (err["y"] - err[LGBM_NAME]).abs()
    by_id = err.groupby("unique_id")["abs"].mean()
    ranked = vol.index.intersection(by_id.index)
    # ponytail: pick a high-volume SKU so the chart is readable, not the best-fit series.
    return str(ranked[0])


def _driver_table(fcst: MLForecast, panel: pd.DataFrame, sample_id: str) -> pd.DataFrame:
    model = next(iter(fcst.models_.values()))
    prepared = fcst.preprocess(
        panel[["unique_id", "ds", "y", *STATIC, *EXOG]],
        static_features=STATIC,
    )
    sample = prepared[prepared["unique_id"].eq(sample_id)].tail(26)
    feature_cols = [c for c in sample.columns if c not in {"unique_id", "ds", "y"}]
    contrib = model.booster_.predict(sample[feature_cols], pred_contrib=True)
    shap = np.asarray(contrib)[:, :-1]
    mean_abs = np.mean(np.abs(shap), axis=0)
    last = shap[-1]
    out = pd.DataFrame(
        {
            "feature": [FEATURE_LABELS.get(c, c) for c in feature_cols],
            "mean_abs_shap": mean_abs,
            "last_week_shap": last,
            "last_week_value": sample[feature_cols].iloc[-1].to_numpy(),
        }
    )
    return out.sort_values("mean_abs_shap", ascending=False).head(12).reset_index(drop=True)


def _promo_whatif(fcst: MLForecast, panel: pd.DataFrame, sample_id: str) -> pd.DataFrame:
    series = panel[panel["unique_id"].eq(sample_id)].sort_values("ds")
    cutoff = series["ds"].iloc[-HORIZON - 1]
    history = panel[panel["ds"] <= cutoff]
    future = panel[panel["ds"] > cutoff][["unique_id", "ds", *EXOG]].copy()
    if future.empty:
        raise ValueError("no holdout weeks for what-if")
    fcst.fit(history[["unique_id", "ds", "y", *STATIC, *EXOG]], static_features=STATIC)
    base = fcst.predict(h=HORIZON, X_df=future)
    promo = future.copy()
    promo["discount"] = np.clip(promo["discount"] + 0.15, 0, 0.9)
    promo["on_promo"] = 1
    alt = fcst.predict(h=HORIZON, X_df=promo)
    actual = panel.loc[
        panel["ds"].isin(base["ds"]) & panel["unique_id"].isin(base["unique_id"]),
        ["unique_id", "ds", "y"],
    ]
    out = base.merge(alt, on=["unique_id", "ds"], suffixes=("_base", "_promo"))
    out = out.merge(actual, on=["unique_id", "ds"], how="left")
    sample = out[out["unique_id"].eq(sample_id)].copy()
    sample = sample.rename(
        columns={
            f"{LGBM_NAME}_base": "forecast",
            f"{LGBM_NAME}_promo": "forecast_15pct_off",
        }
    )
    return sample[["unique_id", "ds", "y", "forecast", "forecast_15pct_off"]]
