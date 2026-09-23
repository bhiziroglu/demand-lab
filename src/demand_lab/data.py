"""Load Walmart M5, cut to FOODS_1 in California, roll up to weeks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
DEPT = "FOODS_1"
STATE = "CA"
MIN_WEEKS = 120
MIN_MEAN_UNITS = 1.0
PROMO_DISCOUNT = 0.05
CACHE_NAME = "foods1_ca_weekly_v3.parquet"


def load_weekly_panel(data_dir: Path | None = None) -> pd.DataFrame:
    """Return a SKU-store weekly panel with price, promo proxy, SNAP, events."""
    data_dir = data_dir or DATA_DIR
    cache = data_dir / "processed" / CACHE_NAME
    if cache.exists():
        return pd.read_parquet(cache)
    raw = _load_m5_slice(data_dir)
    weekly = _filter_active(_to_weekly(raw))
    cache.parent.mkdir(parents=True, exist_ok=True)
    weekly.to_parquet(cache, index=False)
    return weekly


def _load_m5_slice(data_dir: Path) -> pd.DataFrame:
    from datasetsforecast.m5 import M5

    data_dir.mkdir(parents=True, exist_ok=True)
    M5.download(str(data_dir))
    path = data_dir / "m5" / "datasets"

    cal = pd.read_csv(
        path / "calendar.csv",
        usecols=["date", "wm_yr_wk", "event_name_1", "snap_CA"],
        parse_dates=["date"],
    )
    cal["d"] = "d_" + (np.arange(len(cal)) + 1).astype(str)
    prices = pd.read_csv(
        path / "sell_prices.csv",
        dtype={"store_id": "string", "item_id": "string", "sell_price": "float32"},
    )
    sales = _read_sales(path)
    sales = sales[sales["dept_id"].eq(DEPT) & sales["state_id"].eq(STATE)].copy()
    if sales.empty:
        raise ValueError(f"no M5 rows for {DEPT} / {STATE}")

    id_cols = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
    long = sales.melt(id_vars=id_cols, var_name="d", value_name="y")
    long = long.merge(cal, on="d", how="left")
    long = long.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    long = long.sort_values(["item_id", "store_id", "date"])
    started = long["y"].gt(0).groupby([long["item_id"], long["store_id"]]).transform("cummax")
    long = long[started & long["sell_price"].notna()].copy()
    long["unique_id"] = long["item_id"].astype(str) + "_" + long["store_id"].astype(str)
    long["event"] = long["event_name_1"].notna() & ~long["event_name_1"].astype(str).isin(
        ["nan", "NaN", "None", ""]
    )
    long["snap"] = long["snap_CA"].fillna(0).astype("int8")
    long["item_code"] = long["item_id"].astype("category").cat.codes.astype("int16")
    long["store_code"] = long["store_id"].astype("category").cat.codes.astype("int8")
    long["ds"] = pd.to_datetime(long["date"])
    long["y"] = long["y"].astype("float32")
    long["sell_price"] = long["sell_price"].astype("float32")
    long["event"] = long["event"].astype("int8")
    return long[
        [
            "unique_id",
            "ds",
            "y",
            "sell_price",
            "snap",
            "event",
            "item_id",
            "store_id",
            "item_code",
            "store_code",
        ]
    ]


def _read_sales(path: Path) -> pd.DataFrame:
    train_eval = path / "sales_train_evaluation.csv"
    train_val = path / "sales_train_validation.csv"
    test_eval = path / "sales_test_evaluation.csv"
    src = train_eval if train_eval.exists() else train_val
    if not src.exists():
        raise FileNotFoundError(f"M5 sales file missing in {path}")
    sales = pd.read_csv(src)
    if test_eval.exists() and src.name.endswith("evaluation.csv"):
        extra = pd.read_csv(test_eval)
        keys = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
        sales = sales.merge(extra, on=keys, how="left")
    drop = [c for c in sales.columns if c == "id"]
    return sales.drop(columns=drop, errors="ignore")


def _to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily.copy()
    weekly = (
        daily.groupby(["unique_id", pd.Grouper(key="ds", freq="W-SAT")], observed=True)
        .agg(
            y=("y", "sum"),
            sell_price=("sell_price", "mean"),
            snap=("snap", "max"),
            event=("event", "max"),
            item_id=("item_id", "first"),
            store_id=("store_id", "first"),
            item_code=("item_code", "first"),
            store_code=("store_code", "first"),
        )
        .reset_index()
    )
    weekly = _fill_weeks(weekly)
    # last W-SAT label is a stub: M5 daily history ends mid-week
    weekly = weekly[weekly["ds"] < weekly["ds"].max()].copy()
    weekly["list_price"] = weekly.groupby("unique_id", observed=True)["sell_price"].transform(
        "median"
    )
    weekly["discount"] = (
        (weekly["list_price"] - weekly["sell_price"]) / weekly["list_price"]
    ).clip(lower=0)
    weekly["on_promo"] = (weekly["discount"] > PROMO_DISCOUNT).astype("int8")
    weekly["y"] = weekly["y"].astype("float32")
    return weekly


def _fill_weeks(weekly: pd.DataFrame) -> pd.DataFrame:
    """MLForecast rejects gaps. Zero-fill missing weeks, carry price/static forward."""
    pieces = []
    static = ["item_id", "store_id", "item_code", "store_code"]
    for uid, grp in weekly.groupby("unique_id", observed=True):
        g = grp.sort_values("ds").set_index("ds")
        idx = pd.date_range(g.index.min(), g.index.max(), freq="W-SAT")
        g = g.reindex(idx)
        g["y"] = g["y"].fillna(0.0)
        g["sell_price"] = g["sell_price"].ffill().bfill()
        g["snap"] = g["snap"].fillna(0).astype("int8")
        g["event"] = g["event"].fillna(0).astype("int8")
        for col in static:
            g[col] = g[col].ffill().bfill()
        g["unique_id"] = uid
        pieces.append(g.reset_index().rename(columns={"index": "ds"}))
    return pd.concat(pieces, ignore_index=True)


def _filter_active(weekly: pd.DataFrame) -> pd.DataFrame:
    stats = weekly.groupby("unique_id", observed=True)["y"].agg(["count", "mean"])
    keep = stats[(stats["count"] >= MIN_WEEKS) & (stats["mean"] >= MIN_MEAN_UNITS)].index
    out = weekly[weekly["unique_id"].isin(keep)].copy()
    if out.empty:
        raise ValueError("M5 slice is empty after FOODS_1 / CA / activity filters")
    return out.reset_index(drop=True)
