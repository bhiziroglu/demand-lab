"""M5 slice, backtest, then figures and metrics.json."""

from __future__ import annotations

import json
from pathlib import Path

from demand_lab.data import DEPT, STATE, load_weekly_panel
from demand_lab.forecast import HORIZON, N_WINDOWS, run_backtest
from demand_lab.plots import write_figures

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"


def main() -> None:
    panel = load_weekly_panel()
    bt = run_backtest(panel)
    write_figures(bt, REPORTS / "figures")
    payload = {
        "dataset": "M5 (Walmart)",
        "slice": f"{DEPT} × {STATE} stores, weekly",
        "n_series": int(panel["unique_id"].nunique()),
        "n_weeks": int(panel["ds"].nunique()),
        "date_start": str(panel["ds"].min().date()),
        "date_end": str(panel["ds"].max().date()),
        "horizon_weeks": HORIZON,
        "cv_windows": N_WINDOWS,
        "sample_id": bt.sample_id,
        "promo_share": round(float(panel["on_promo"].mean()), 3),
        "metrics": bt.metrics.to_dict(orient="records"),
        "whatif_lift": _whatif_lift(bt.whatif),
        "pi": {k: round(float(v), 2) for k, v in bt.pi.items()},
        "top_drivers": bt.drivers.head(8).to_dict(orient="records"),
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "metrics.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(json.dumps(payload, indent=2, default=str))


def _whatif_lift(whatif) -> float:
    base = float(whatif["forecast"].sum())
    alt = float(whatif["forecast_15pct_off"].sum())
    if base == 0:
        return 0.0
    return round((alt - base) / base, 3)


if __name__ == "__main__":
    main()
