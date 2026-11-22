"""One-circuit probe job on ibm_fez (frozen registration hw_prereg_daily2024.json): test row 0, 6 qubits, 1 layer, seed 0, 2000 shots, no mitigation. Run with .venv-qiskit from qlstm_copper/."""
import itertools, json, sys, numpy as np
from qiskit import transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
sys.path.insert(0, "hw"); import qiskit_qrc as Q
n, L, SH = 6, 1, 2000
z = np.load("hw/daily2024_rebuild.npz"); x = z["inputs_te"][0]; ex = Q.exact_features(z["inputs_te"][:1], 0, L, n=n)[0]
svc = QiskitRuntimeService(); B = svc.backend("ibm_fez")
qc = Q.circuit(x, Q.angles(n, 0, L), n); qc.measure_all()
t = transpile(qc, B, optimization_level=3, seed_transpiler=1)
print("transpiled depth", t.depth(), "2q gates", sum(v for k, v in t.count_ops().items() if k in ("cz", "ecr", "cx")), "layout", t.layout.final_index_layout() if t.layout else None, flush=True)
job = SamplerV2(mode=B).run([t], shots=SH); print("JOB ID", job.job_id(), flush=True)
res = job.result(); counts = res[0].data.meas.get_counts()
P = np.zeros(2 ** n)
for k, v in counts.items(): P[int(k, 2)] = v / SH
zq = [1 - 2 * ((np.arange(2 ** n) >> q) & 1) for q in range(n)]
f = np.array([P @ zq[i] for i in range(n)] + [P @ (zq[i] * zq[j]) for i, j in itertools.combinations(range(n), 2)])
print("status", job.status(), "| usage:", job.usage() if hasattr(job, "usage") else None)
print("mean |hw - exact| per feature:", round(float(np.abs(f - ex).mean()), 4), "max", round(float(np.abs(f - ex).max()), 4))
print("backend", B.name, "calibration/processor:", getattr(B, "processor_type", None))
np.savez("hw/probe_result.npz", f=f, exact=ex, job_id=job.job_id(), shots=SH)
