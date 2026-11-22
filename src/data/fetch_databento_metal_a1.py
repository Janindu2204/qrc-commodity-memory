"""Copy of fetch_databento_copper.py for gold/silver (registration PROGRESS/34): file stem taken from the symbol so the copper file is never overwritten.
"""


import argparse
import os
import sys
from pathlib import Path

DATASET = "GLBX.MDP3"
OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "databento"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--symbol", required=True, help="GC.v.0 or SI.v.0")
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
    stem = f"{args.symbol.split('.')[0]}_v0_{args.schema}_{args.start}_{args.end}"
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
