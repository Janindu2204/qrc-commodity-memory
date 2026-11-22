"""Download COMEX copper (HG) bars from Databento and save raw DBN + CSV.

Prices the request first (free metadata call) and aborts if it exceeds
--max-cost, so the free credit cannot be spent by accident. The key comes from
the DATABENTO_API_KEY environment variable only.

    python -m src.data.fetch_databento_copper --start 2025-01-01 --end 2026-01-01

Outputs in data/databento/ (UTC timestamps, continuous contract rolled on
volume; the raw .dbn.zst is kept unchanged, the CSV is a plain export of it):
    HG_v0_<schema>_<start>_<end>.dbn.zst
    HG_v0_<schema>_<start>_<end>.csv

Licensed data: do not commit or redistribute the files; share this script and
the exact request instead. Second bars exist only for seconds with a trade.
"""

import argparse
import os
import sys
from pathlib import Path

DATASET = "GLBX.MDP3"
OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "databento"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--symbol", default="HG.v.0")
    p.add_argument("--stype", default="continuous")
    p.add_argument("--schema", default="ohlcv-1s")
    p.add_argument("--start", required=True, help="UTC date YYYY-MM-DD")
    p.add_argument("--end", required=True, help="UTC date YYYY-MM-DD, exclusive")
    p.add_argument("--max-cost", type=float, default=15.0, help="USD cap")
    args = p.parse_args()

    if not os.environ.get("DATABENTO_API_KEY"):
        print("Set DATABENTO_API_KEY in the environment first.", file=sys.stderr)
        return 1
    import databento as db

    client = db.Historical()
    req = dict(
        dataset=DATASET,
        symbols=[args.symbol],
        stype_in=args.stype,
        schema=args.schema,
        start=args.start,
        end=args.end,
    )
    cost = client.metadata.get_cost(**req)
    print(f"Estimated cost: ${cost:.4f} (cap ${args.max_cost:.2f})")
    if cost > args.max_cost:
        print("Over the cap; nothing downloaded.", file=sys.stderr)
        return 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"HG_v0_{args.schema}_{args.start}_{args.end}"
    dbn_path = OUT_DIR / f"{stem}.dbn.zst"
    csv_path = OUT_DIR / f"{stem}.csv"

    store = client.timeseries.get_range(path=str(dbn_path), **req)
    df = store.to_df()  # index = ts_event in UTC
    df.index.name = "timestamp_utc"
    df.to_csv(csv_path)
    print(f"Saved {len(df):,} rows")
    print(f"  raw: {dbn_path}")
    print(f"  csv: {csv_path}")
    print(f"  span: {df.index.min()} -> {df.index.max()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
