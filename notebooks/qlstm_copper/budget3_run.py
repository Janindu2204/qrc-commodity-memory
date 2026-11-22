"""long_prereg_budget3.json. Usage: python budget3_run.py   Writes budget_out/budget3.json and prints the report.
Part A: second dataset (budget3_lib): 64 protocols (best and typical), Shapley on the 'best' statistic, month-block null (B = 1000, seeds 44000+b)
for control, each single selection pitfall and ALL5. Part B: 'typical researcher' statistic on copper (budget_lib tables unchanged; mean of the log
ratio over the open options instead of the minimum) for control, each single pitfall, ALL5 and all six."""
import json, time
from pathlib import Path
import numpy as np
import budget3_lib as b3, budget_lib as bl
HERE = Path(__file__).parent; OUT = HERE / "budget_out"; t0 = time.time(); B = 1000
TT = {}
for a in b3.ASSETS: TT[a] = b3.tables(b3.load_asset(a)); print(a, "refs (best, worst by validation):", b3.refs(TT[a]), "quarters", TT[a]["quarters"], f"{time.time() - t0:.0f}s", flush=True)
P = b3.all_protocols(); key = lambda f: tuple(f[k] for k in b3.FACTORS)
best = {key(f): b3.protocol(TT, None, f, "best") for f in P}; typ = {key(f): b3.protocol(TT, None, f, "typical") for f in P}
ph = bl.shapley({k: float(np.log(v)) for k, v in best.items()}); ph = dict(zip(b3.FACTORS, ph.values()))
on = lambda **kw: tuple(bool(kw.get(k, False)) for k in b3.FACTORS)
SEL = {"control": on(), **{f: on(**{f: True}) for f in b3.FACTORS[:5]}, "ALL5": on(SEED=1, WINDOW=1, ASSET=1, LOSS=1, QMODEL=1)}
allm = np.unique(np.concatenate([TT[a]["months"] for a in b3.ASSETS])); null = {n: [] for n in SEL}
for b in range(B):
    coin = np.random.default_rng(44000 + b).random(len(allm)) < 0.5; flips = {a: coin[np.searchsorted(allm, TT[a]["months"])] for a in b3.ASSETS}
    for n, k in SEL.items(): null[n].append(b3.protocol(TT, flips, dict(zip(b3.FACTORS, k)), "best"))
    if b % 200 == 0: print("null", b, f"{time.time() - t0:.0f}s", flush=True)

# Part B: copper typical
def typical_grid(W, wy, rs, f):
    vals = []
    for q in (bl.QUANT if f["QMODEL"] else ["DOM"]):
        for L in (bl.LOSSES if f["LOSS"] else ["MSE"]):
            refs = {Y: (bl.WEAK if f["REF"] else rs[L][Y]) for Y in bl.TEST}; subs = range(bl.FULL) if f["SEED"] else [bl.FULL]
            Qw = np.empty((len(subs), len(wy))); Rw = np.empty(len(wy))
            for Y in bl.TEST:
                c = wy == Y; Qw[:, c] = W[L][q][list(subs)][:, c]; Rw[c] = W[L][refs[Y]][bl.FULL][c]
            for ys in ([[Y] for Y in bl.TEST] if f["WINDOW"] else [bl.TEST]):
                c = np.isin(wy, ys); vals += list(np.log(Qw[:, c].sum(1) / Rw[c].sum()))
    return float(np.mean(vals))
tabs = {}
for g in bl.GRIDS:
    W, cnt, wy, ids = bl.weekly_tables(g); tabs[g] = (W, wy, bl.ref_selection(W, wy))
ck = lambda **kw: {k: bool(kw.get(k, False)) for k in bl.FACTORS}
CP = {"control": ck(), **{f: ck(**{f: True}) for f in bl.FACTORS}, "ALL5": ck(SEED=1, WINDOW=1, GRID=1, LOSS=1, QMODEL=1), "ALL6": {k: True for k in bl.FACTORS}}
cop_typ = {n: float(np.exp(np.mean([typical_grid(*tabs[g], f) for g in bl.GRIDS]))) for n, f in CP.items()}   # GRID pitfall 'typical' = random grid = mean over grids
json.dump(dict(best={str(k): v for k, v in best.items()}, typical={str(k): v for k, v in typ.items()}, shapley=ph, null=null, copper_typical=cop_typ,
               refs={a: b3.refs(TT[a]) for a in b3.ASSETS}), open(OUT / "budget3.json", "w"), indent=1)

A = lambda r: 100 * (1 - r)
print("\n== Part A: 11 commodity futures, hourly, QLSTMT and five other quantum models ==")
print(f"  control ratio {best[SEL['control']]:.4f}  apparent advantage {A(best[SEL['control']]):+.2f}%")
for n in list(SEL)[1:] + ["ALL6"]:
    k = (True,) * 6 if n == "ALL6" else SEL[n]; print(f"  {n:7s} best {A(best[k]):+6.2f}%   typical {A(typ[k]):+6.2f}%")
tot = sum(ph.values()); print("  Shapley shares (best):", {k: round(100 * v / tot, 1) for k, v in sorted(ph.items(), key=lambda kv: kv[1])})
c0 = np.log(best[SEL["control"]]); cn = np.log(np.array(null["control"]))
print("  month-block null, gain over control (log points): observed | null median | null 95th | verdict")
for n in list(SEL)[1:]:
    go = 100 * (c0 - np.log(best[SEL[n]])); gn = 100 * (cn - np.log(np.array(null[n]))); q95 = np.quantile(gn, 0.95)
    print(f"    {n:7s} {go:6.2f} | {np.median(gn):6.2f} | {q95:6.2f} | {'beyond chance' if go > q95 else 'within chance'}")
a5 = A(best[SEL["ALL5"]]); top2 = [k for k, _ in sorted(ph.items(), key=lambda kv: kv[1])[:2]]
print("  replication rule:", "REPLICATES" if (a5 > 5 and ({"LOSS", "WINDOW"} & set(top2))) else "DOES NOT REPLICATE", f"(ALL5 best {a5:+.2f}%, top-2 Shapley factors {top2})")
print("\n== Part B: copper, typical researcher (random option) ==")
for n, r in cop_typ.items(): print(f"  {n:7s} typical apparent advantage {A(r):+.2f}%")
print(f"\ndone {time.time() - t0:.0f}s")
