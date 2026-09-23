"""Seaborn charts with 80% residual bands."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from demand_lab.forecast import LGBM_NAME, NAIVE_NAME, Backtest

PALETTE = sns.color_palette("colorblind")
ACTUAL = "#111111"
FORECAST = PALETTE[0]
NAIVE = "#6b6b6b"
PROMO = PALETTE[1]
BAND = PALETTE[0]


def write_figures(bt: Backtest, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _style()
    _forecast_vs_actual(bt, out_dir / "forecast_vs_actual.png")
    _store_rollup(bt, out_dir / "store_rollup.png")
    _error_by_promo(bt, out_dir / "error_by_promo.png")
    _drivers(bt, out_dir / "drivers.png")
    _whatif(bt, out_dir / "whatif_promo.png")


def _style() -> None:
    sns.set_theme(style="whitegrid", palette="colorblind", context="notebook")
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 160,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.frameon": False,
        }
    )


def _band(ax, ds, yhat, lo: float, hi: float, label: str = "80% band") -> None:
    lower = np.clip(yhat + lo, 0, None)
    upper = yhat + hi
    ax.fill_between(ds, lower, upper, color=BAND, alpha=0.22, linewidth=0, label=label, zorder=1)


def _forecast_vs_actual(bt: Backtest, path: Path) -> None:
    uid = bt.sample_id
    cutoff = bt.lgbm_cv["cutoff"].max()
    hist = bt.panel[bt.panel["unique_id"].eq(uid)].sort_values("ds").tail(40)
    fold = (
        bt.lgbm_cv[bt.lgbm_cv["unique_id"].eq(uid) & bt.lgbm_cv["cutoff"].eq(cutoff)]
        .sort_values("ds")
    )
    naive = (
        bt.naive_cv[bt.naive_cv["unique_id"].eq(uid) & bt.naive_cv["cutoff"].eq(cutoff)]
        .sort_values("ds")
    )
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    sns.lineplot(data=hist, x="ds", y="y", ax=ax, color=ACTUAL, lw=2.2, label="actual")
    _band(ax, fold["ds"], fold[LGBM_NAME].to_numpy(), bt.pi["sku_lo"], bt.pi["sku_hi"])
    sns.lineplot(data=fold, x="ds", y=LGBM_NAME, ax=ax, color=FORECAST, lw=2.2, label="LightGBM")
    sns.lineplot(
        data=naive, x="ds", y=NAIVE_NAME, ax=ax, color=NAIVE, lw=1.6, linestyle="--", label="seasonal naive"
    )
    ax.set_title(f"Weekly units · {uid}")
    ax.set_xlabel("")
    ax.set_ylabel("units")
    ax.legend(loc="upper left")
    sns.despine()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _store_rollup(bt: Backtest, path: Path) -> None:
    uid = bt.sample_id
    store = bt.panel.loc[bt.panel["unique_id"].eq(uid), "store_id"].iloc[0]
    cutoff = bt.lgbm_cv["cutoff"].max()
    hist = (
        bt.panel[bt.panel["store_id"].eq(store)]
        .groupby("ds", as_index=False)["y"]
        .sum()
        .sort_values("ds")
        .tail(40)
    )
    fold = bt.lgbm_cv[bt.lgbm_cv["cutoff"].eq(cutoff)].merge(
        bt.panel[["unique_id", "ds", "store_id"]], on=["unique_id", "ds"]
    )
    fc = (
        fold[fold["store_id"].eq(store)]
        .groupby("ds", as_index=False)[LGBM_NAME]
        .sum()
        .sort_values("ds")
    )
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    sns.lineplot(data=hist, x="ds", y="y", ax=ax, color=ACTUAL, lw=2.2, label="actual")
    _band(ax, fc["ds"], fc[LGBM_NAME].to_numpy(), bt.pi["store_lo"], bt.pi["store_hi"])
    sns.lineplot(
        data=fc, x="ds", y=LGBM_NAME, ax=ax, color=FORECAST, lw=2.2, label="bottom-up LightGBM"
    )
    ax.set_title(f"Store rollup · {store} · FOODS_1 (sum of SKUs)")
    ax.set_xlabel("")
    ax.set_ylabel("units")
    ax.legend(loc="upper left")
    sns.despine()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _error_by_promo(bt: Backtest, path: Path) -> None:
    m = bt.metrics[bt.metrics["bucket"].isin(["promo", "no_promo"])].copy()
    m["weeks"] = m["bucket"].map(
        {"no_promo": "regular weeks", "promo": "promo weeks (>5% off)"}
    )
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    sns.barplot(
        data=m,
        x="weeks",
        y="mase",
        hue="model",
        order=["regular weeks", "promo weeks (>5% off)"],
        hue_order=["LightGBM", "SeasonalNaive"],
        palette={"LightGBM": FORECAST, "SeasonalNaive": NAIVE},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("MASE")
    ax.set_title("Holdout MASE, regular weeks vs weeks with a >5% discount")
    ax.legend(title="")
    sns.despine()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _drivers(bt: Backtest, path: Path) -> None:
    d = bt.drivers.sort_values("mean_abs_shap", ascending=False)
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    sns.barplot(
        data=d,
        y="feature",
        x="mean_abs_shap",
        color=FORECAST,
        order=d["feature"],
        ax=ax,
    )
    ax.set_xlabel("mean |SHAP| over last 26 weeks")
    ax.set_ylabel("")
    ax.set_title(f"Why the forecast moved · {bt.sample_id}")
    sns.despine()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _whatif(bt: Backtest, path: Path) -> None:
    w = bt.whatif.sort_values("ds")
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    sns.lineplot(data=w, x="ds", y="y", ax=ax, color=ACTUAL, lw=2.2, label="actual")
    _band(ax, w["ds"], w["forecast"].to_numpy(), bt.pi["sku_lo"], bt.pi["sku_hi"])
    sns.lineplot(data=w, x="ds", y="forecast", ax=ax, color=FORECAST, lw=2.2, label="forecast")
    sns.lineplot(
        data=w, x="ds", y="forecast_15pct_off", ax=ax, color=PROMO, lw=2.2, label="15% price cut"
    )
    ax.set_title(f"Promo what-if · {bt.sample_id}")
    ax.set_xlabel("")
    ax.set_ylabel("units")
    ax.legend(loc="upper left")
    sns.despine()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
