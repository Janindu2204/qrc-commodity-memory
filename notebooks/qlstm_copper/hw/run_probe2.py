"""Probe 2 (frozen registration hw_prereg_daily2024.json): test rows 1-10 as ONE job of 10 circuits on ibm_fez, 2000 shots each, no mitigation. Separates fixed cost per job from cost per circuit."""
import itertools, sys, numpy as np
from qiskit import transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
sys.path.insert(0, "hw"); import qiskit_qrc as Q
n, L, SH, R = 6, 1, 2000, list(range(1, 11))
z = np.load("hw/daily2024_rebuild.npz"); X = z["inputs_te"][R]; ex = Q.exact_features(X, 0, L, n=n)
B = QiskitRuntimeService().backend("ibm_fez"); ang = Q.angles(n, 0, L); qcs = []
for x in X:
    qc = Q.circuit(x, ang, n); qc.measure_all(); qcs.append(qc)
ts = transpile(qcs, B, optimization_level=3, seed_transpiler=1)
job = SamplerV2(mode=B).run(ts, shots=SH); print("JOB ID", job.job_id(), flush=True)
res = job.result(); zq = [1 - 2 * ((np.arange(2 ** n) >> q) & 1) for q in range(n)]; F = []
for i in range(len(R)):
    c = res[i].data.meas.get_counts(); P = np.zeros(2 ** n)
    for k, v in c.items(): P[int(k, 2)] = v / SH
    F.append([P @ zq[a] for a in range(n)] + [P @ (zq[a] * zq[b]) for a, b in itertools.combinations(range(n), 2)])
F = np.array(F); print("status", job.status(), "| mean |hw-exact| per feature", round(float(np.abs(F - ex).mean()), 4), "max", round(float(np.abs(F - ex).max()), 4))
np.savez("hw/probe2_result.npz", f=F, exact=ex, rows=np.array(R), job_id=job.job_id(), shots=SH)
