"""Build the high-frequency Databento copper datasets for long_prereg_hf.json from the 1-minute file (HG.v.0, 2010-06-07 .. 2026-10-17).
Usage: python hf_build.py      Writes data/hf/F.npz (5-minute bins; H and M come from hf_build.py). Scores nothing.

Per bin (start time s, width w): ret = ln(last close in bin / last close in the previous bin) if the previous observed bin is adjacent
(gap == w) or across the routine daily break (gap == w + 60 min and the previous bin ends 21:00 or 22:00 UTC) and the contract (instrument_id) is the same;
lrv = ln(sum of squared log differences of consecutive observed 1-minute closes inside the bin, same contract) ; lvol = ln(1 + volume);
n = number of 1-minute bars; target-ok requires n >= MIN_N. hs, hc = sin/cos of hour of day (UTC) of the bin start.
"""
import glob
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]; D = ROOT / "data" / "databento"; OUT = ROOT / "data" / "hf"; OUT.mkdir(exist_ok=True)
d = pd.read_csv(glob.glob(str(D / "HG_v0_ohlcv-1m_*.csv"))[0], usecols=["timestamp_utc", "instrument_id", "close", "volume"], parse_dates=["timestamp_utc"]).sort_values("timestamp_utc").reset_index(drop=True)
print("1-minute bars:", len(d), d.timestamp_utc.min(), d.timestamp_utc.max())
lc = np.log(d.close.values); inst = d.instrument_id.values
dr = np.r_[np.nan, np.diff(lc)]; samecon = np.r_[False, inst[1:] == inst[:-1]]


def build(width_min, min_n, name):
    w = pd.Timedelta(minutes=width_min)
    b = d.timestamp_utc.dt.floor(f"{width_min}min")
    same = (b.values == np.r_[b.values[:1], b.values[:-1]]) & samecon
    sq = np.where(same, dr ** 2, 0.0)
    g = pd.DataFrame({"b": b, "sq": sq, "vol": d.volume.values, "lc": lc, "inst": inst, "one": 1})
    a = g.groupby("b").agg(rv=("sq", "sum"), vol=("vol", "sum"), n=("one", "sum"), lc=("lc", "last"), inst=("inst", "last"))
    s = a.index.to_series()
    gap = s.diff()
    prev_end = (s - w).dt.hour * 60 + (s - w).dt.minute      # not used directly; see below
    pe = (s.shift() + w); pe_min = pe.dt.hour * 60 + pe.dt.minute
    adj = (gap == w) | ((gap == w + pd.Timedelta(minutes=60)) & pe_min.isin([21 * 60, 22 * 60]))
    adj &= (a.inst.values == a.inst.shift().values)
    ret = (a.lc - a.lc.shift()).where(adj)
    hod = (s.dt.hour + s.dt.minute / 60).values
    out = dict(t=s.values.astype("datetime64[ns]").astype(np.int64), ret=ret.values, lrv=np.log(a.rv.values + 1e-12), lvol=np.log1p(a.vol.values), n=a.n.values,
               hs=np.sin(2 * np.pi * hod / 24), hc=np.cos(2 * np.pi * hod / 24), hodbin=s.dt.hour.values, adj=adj.values, rvpos=(a.rv.values > 0) & (a.n.values >= min_n))
    np.savez(OUT / f"{name}.npz", **out)
    print(f"{name}: bins {len(a)} | adjacent {int(adj.sum())} | rv>0 and n>={min_n}: {int(out['rvpos'].sum())} | median 1m bars/bin {int(np.median(a.n.values))}")


build(5, 2, "F")
