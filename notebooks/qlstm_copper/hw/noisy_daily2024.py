"""Qiskit venv. (1) verify 6-qubit 1-layer port against numpy features F0 on all rows (test rows); (2) sample test features on ideal and FakeFez-noise simulators at several shot counts; (3) transpiled cost. Writes hw/daily2024_noisy.npz."""
import itertools, numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeFez
import qiskit_qrc as Q
z = np.load("hw/daily2024_rebuild.npz"); Xte, te, F0 = z["inputs_te"], z["te"], z["F0"][z["te"]]; n, L, seed = 6, 1, 0
ang = Q.angles(n, seed, L)
ex = Q.exact_features(Xte, seed, L, n=n); print("port check on 252 test rows, max |qiskit-numpy|:", np.abs(ex - F0).max())
def feats(counts, shots):
    P = np.zeros(2 ** n)
    for k, v in counts.items(): P[int(k, 2)] = v / shots
    zq = [1 - 2 * ((np.arange(2 ** n) >> q) & 1) for q in range(n)]
    return np.array([P @ zq[i] for i in range(n)] + [P @ (zq[i] * zq[j]) for i, j in itertools.combinations(range(n), 2)])
B = FakeFez(); sims = {"ideal": AerSimulator(), "fakefez": AerSimulator.from_backend(B)}
qcs = []
for x in Xte:
    qc = Q.circuit(x, ang, n); qc.measure_all(); qcs.append(qc)
t = transpile(qcs[0], B, optimization_level=3, seed_transpiler=1, scheduling_method="alap")
print("6q 1-layer transpiled: depth", t.depth(), "2q gates", sum(v for k, v in t.count_ops().items() if k in ("cz", "ecr", "cx")), "duration us", round(t.duration * B.dt * 1e6, 1))
out = {"exact": ex}
for name, sim in sims.items():
    tq_ = transpile(qcs, B if name == "fakefez" else sim, optimization_level=3, seed_transpiler=1)
    for shots in (500, 2000, 8000):
        res = sim.run(tq_, shots=shots, seed_simulator=11).result()
        out[f"{name}_{shots}"] = np.array([feats(res.get_counts(i), shots) for i in range(len(qcs))])
        print(name, shots, "mean |err| per feature", round(float(np.abs(out[f'{name}_{shots}'] - ex).mean()), 4))
np.savez("hw/daily2024_noisy.npz", **out)
