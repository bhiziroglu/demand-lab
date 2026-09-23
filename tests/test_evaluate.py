from demand_lab.evaluate import _seasonal_scale, scaled_errors
from demand_lab.forecast import residual_quantiles
import pandas as pd


def test_seasonal_scale_is_mean_abs_seasonal_diff():
    y = pd.Series([10.0] * 52 + [12.0] * 52).to_numpy()
    assert abs(_seasonal_scale(y, 52) - 2.0) < 1e-9


def test_mase_is_one_when_forecast_matches_seasonal_naive_scale():
    train = pd.DataFrame(
        {
            "unique_id": ["a"] * 60,
            "ds": pd.date_range("2020-01-04", periods=60, freq="W-SAT"),
            "y": [1.0] * 52 + [3.0] * 8,
        }
    )
    actual = pd.DataFrame(
        {
            "unique_id": ["a"] * 4,
            "ds": pd.date_range("2021-02-27", periods=4, freq="W-SAT"),
            "y": [3.0, 3.0, 3.0, 3.0],
        }
    )
    pred = actual.copy()
    pred["model"] = 1.0
    scores = scaled_errors(actual, pred, train, "model", season=52)
    # train seasonal diffs are 2.0 (the last 8 of the first 52 vs the next 8).
    # |3-1| / 2 = 1.0
    assert abs(float(scores["mase"].iloc[0]) - 1.0) < 1e-6


def test_residual_quantiles_are_10_90():
    resid = pd.Series(list(range(1, 11)), dtype=float)
    lo, hi = residual_quantiles(resid)
    assert lo == resid.quantile(0.1)
    assert hi == resid.quantile(0.9)
