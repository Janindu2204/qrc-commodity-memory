"""Free dry run on a fake IBM backend (no account, no QPU time): transpiled size, estimated QPU seconds, and noisy-vs-exact feature error. Run with .venv-qiskit."""
import sys, itertools, numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeFez
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeFez
import qiskit_qrc as Q

REP_DELAY = 250e-6                                   # IBM default rep_delay (assumption; check the live backend)
z = np.load("hw/exact_ref.npz"); X = z["X"]; n = 8
for name, B in (("FakeFez (Heron)", FakeFez()), ("FakeSherbrooke (Eagle)", FakeSherbrooke())):
    for L in (1, 2):
        ang = Q.angles(n, 0, L); qc = Q.circuit(X[0], ang); qc.measure_all()
        t = transpile(qc, B, optimization_level=3, seed_transpiler=1, scheduling_method="alap")
        ops = t.count_ops(); two = sum(v for k, v in ops.items() if k in ("cz", "ecr", "cx"))
        dur = (t.duration or 0) * B.dt
        print(f"{name} layers {L}: depth {t.depth()} two-qubit gates {two} circuit duration {dur*1e6:.1f} us -> per shot ~{(dur+REP_DELAY)*1e6:.0f} us, 1000 shots ~{1000*(dur+REP_DELAY):.2f} s")
# noisy simulation error (layers 2) on FakeFez noise, 20 inputs, several shot counts
B = FakeFez(); sim = AerSimulator.from_backend(B); L = 2; ang = Q.angles(n, 0, L); ex = z["seed0_L2"]
def feats(counts, shots):
    P = np.zeros(2 ** n)
    for k, v in counts.items(): P[int(k, 2)] = v / shots
    bit = lambda q: (np.arange(2 ** n) >> q) & 1                      # qiskit: rightmost char = qubit 0
    zq = [1 - 2 * bit(q) for q in range(n)]
    return np.array([P @ zq[i] for i in range(n)] + [P @ (zq[i] * zq[j]) for i, j in itertools.combinations(range(n), 2)])
for shots in (500, 2000):
    err = []
    for i, x in enumerate(X):
        qc = Q.circuit(x, ang); qc.measure_all(); t = transpile(qc, B, optimization_level=3, seed_transpiler=1)
        # measured layout may permute qubits: map back through the final layout via the measure register (measure_all keeps clbit order)
        c = sim.run(t, shots=shots, seed_simulator=i).result().get_counts()
        err.append(np.abs(feats(c, shots) - ex[i]))
    err = np.array(err); print(f"noisy FakeFez shots {shots}: mean |error| per feature {err.mean():.3f}, max {err.max():.3f}; feature std across inputs {ex.std():.3f}")
# ideal sampling noise only, same shots
for shots in (500, 2000):
    ideal = AerSimulator(); err = []
    for i, x in enumerate(X):
        qc = Q.circuit(x, ang); qc.measure_all(); c = ideal.run(qc, shots=shots, seed_simulator=i).result().get_counts(); err.append(np.abs(feats(c, shots) - ex[i]))
    print(f"ideal sampling only shots {shots}: mean |error| {np.mean(err):.3f}")
