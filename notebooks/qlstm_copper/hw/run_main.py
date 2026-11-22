"""Main run (hw_prereg_daily2024.json + addendum1). Usage: python hw/run_main.py JOB   JOB = 0,1,2,3 (63 shuffled test rows each) or repeat (rows 0,12,...,228).
ibm_fez, 2000 shots, optimization_level 3, seed_transpiler 1, no mitigation. Saves hw/main_job_<JOB>.npz. Run with .venv-qiskit from qlstm_copper/."""
import itertools, sys, numpy as np
from qiskit import transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
sys.path.insert(0, "hw"); import qiskit_qrc as Q
JOB = sys.argv[1]; n, L, SH = 6, 1, 2000
z = np.load("hw/daily2024_rebuild.npz"); X = z["inputs_te"]; N = len(X)
perm = np.random.default_rng(2024).permutation(N)
rows = np.array(sorted(range(0, N, 12))[:20]) if JOB == "repeat" else np.sort(perm[int(JOB) * 63:(int(JOB) + 1) * 63]) if False else (np.array(sorted(range(0, N, 12))[:20]) if JOB == "repeat" else perm[int(JOB) * 63:(int(JOB) + 1) * 63])
print("job", JOB, "rows", len(rows), flush=True)
B = QiskitRuntimeService().backend("ibm_fez"); ang = Q.angles(n, 0, L); qcs = []
for x in X[rows]:
    qc = Q.circuit(x, ang, n); qc.measure_all(); qcs.append(qc)
ts = transpile(qcs, B, optimization_level=3, seed_transpiler=1)
job = SamplerV2(mode=B).run(ts, shots=SH); print("JOB ID", job.job_id(), flush=True)
res = job.result(); zq = [1 - 2 * ((np.arange(2 ** n) >> q) & 1) for q in range(n)]; F = []
for i in range(len(rows)):
    c = res[i].data.meas.get_counts(); P = np.zeros(2 ** n)
    for k, v in c.items(): P[int(k, 2)] = v / SH
    F.append([P @ zq[a] for a in range(n)] + [P @ (zq[a] * zq[b]) for a, b in itertools.combinations(range(n), 2)])
F = np.array(F); ex = z["F0"][z["te"]][rows]
print("status", job.status(), "| mean |hw-exact| per feature", round(float(np.abs(F - ex).mean()), 4), "max", round(float(np.abs(F - ex).max()), 4))
np.savez(f"hw/main_job_{JOB}.npz", f=F, rows=rows, job_id=job.job_id(), shots=SH)
