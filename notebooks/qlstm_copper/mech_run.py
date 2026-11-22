"""Part 4 (long_prereg_mech.json): capacity profiles. Usage: python mech_run.py FAMILY   FAMILY = DOM | DOMG1 | ISING | ESN | SAS | ESNB | SASB.
i.i.d. uniform input (mech_lib.inputs(0)), no copper data. Writes mech_out/cap_<FAMILY>.json. Nothing here is a forecast score."""
import sys, json, time
from pathlib import Path
import numpy as np
import dom_lib as dl, dom2_lib as d2, dom3_lib as d3, dom4_lib as d4, mech_lib as ml

FAM = sys.argv[1]; HERE = Path(__file__).parent; u = ml.inputs(0); zs = (u - u[ml.WASH:ml.FIT].mean()) / u[ml.WASH:ml.FIT].std()
cc = np.sqrt(u * (1 - u)); Z = np.column_stack([zs, (cc - cc[ml.WASH:ml.FIT].mean()) / cc[ml.WASH:ml.FIT].std()])
runs = []
if FAM in ("DOM", "DOMG1", "ISING"):
    for s in range(10): runs.append(({"seed": s}, (lambda s=s: dl.features(u, s, "G3" if FAM == "DOM" else "G1") if FAM != "ISING" else d2.ising_features(u, s))))
else:
    grid = d3.ESN_GRID if FAM in ("ESN", "ESNB") else d3.SAS_GRID; seeds = range(5) if FAM in ("ESN", "SAS") else range(3)
    fn = {"ESN": lambda s, c: dl.esn1(zs, s, leak=c["leak"], rho=c["rho"]), "SAS": lambda s, c: d3.sas(u, s, leak=c["leak"], gain=c["gain"]),
          "ESNB": lambda s, c: d4.esn_big(Z, s, leak=c["leak"], rho=c["rho"]), "SASB": lambda s, c: d4.sas_big(u, s, leak=c["leak"], gain=c["gain"])}[FAM]
    for c in grid:
        for s in seeds: runs.append(({"seed": s, **c}, (lambda s=s, c=c: fn(s, c))))
out = []; t0 = time.time()
for meta, f in runs:
    r = ml.capacity(f(), u); out.append(dict(meta, MC=r["MC"], C2=r["C2"], H90=ml.horizon(r["MF"]), MF=[round(float(x), 5) for x in r["MF"]],
                                            C2_sq=[round(float(x), 5) for x in r["C2_sq"]], C2_cross=[round(float(x), 5) for x in r["C2_cross"]]))
    print(FAM, meta, f"MC {r['MC']:.2f} C2 {r['C2']:.2f} H90 {out[-1]['H90']}", f"{time.time() - t0:.0f}s", flush=True)
(HERE / "mech_out").mkdir(exist_ok=True); json.dump(out, open(HERE / "mech_out" / f"cap_{FAM}.json", "w"))
