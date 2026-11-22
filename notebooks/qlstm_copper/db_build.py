"""Build daily copper bars from the Databento hourly file by CME trade date (18:00 ET roll). Writes data/daily_databento/HG_db_1d_range.csv and _rv.csv.
range file: daily high/low = max/min of hourly highs/lows. rv file: high/low synthesised so that ln(H/L)^2/(4 ln 2) equals the sum of squared hourly log returns
(no cross-gap returns), letting the same long_core.prepare estimator run on a realised-variance target. Days with fewer than 6 hourly bars are dropped."""
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
h = pd.read_csv(ROOT / "data/databento/HG_v0_ohlcv-1h_2010-06-07_2026-10-17.csv", parse_dates=["timestamp_utc"])
h["et"] = h.timestamp_utc.dt.tz_convert("America/New_York"); h["td"] = (h.et + pd.Timedelta(hours=6)).dt.date
h = h.sort_values("timestamp_utc"); h["lr"] = np.log(h.close).diff(); h.loc[h.timestamp_utc.diff() > pd.Timedelta(hours=3), "lr"] = np.nan
g = h.groupby("td")
d = pd.DataFrame({"open": g.open.first(), "high": g.high.max(), "low": g.low.min(), "close": g.close.last(), "volume": g.volume.sum(), "nbars": g.size(), "rv": (h.lr ** 2).groupby(h.td).sum(min_count=1)})
d.index = pd.to_datetime(d.index); d = d[(d.nbars >= 6) & (d.index <= "2026-10-09") & (d.index.dayofweek < 5)]
d["gap_clean"] = 0.0; d["is_ambiguous"] = False
d.index.name = "date"; d.drop(columns="rv").to_csv(ROOT / "data/daily_databento/HG_db_1d_range.csv")
r = d.copy(); r["low"] = 1.0; r["high"] = np.exp(np.sqrt(4 * np.log(2) * r.rv.where(r.rv > 0)))
r.drop(columns="rv").to_csv(ROOT / "data/daily_databento/HG_db_1d_rv.csv")
print(len(d), d.index.min().date(), d.index.max().date(), "zero-range days", int((d.high == d.low).sum()), "rv<=0", int((d.rv <= 0).sum()), "dropped <6 bars handled")
