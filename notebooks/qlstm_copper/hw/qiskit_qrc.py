"""Qiskit port of the project's 8-qubit QRC (hourly_databento_qrc.reservoir_features): RY(arctan z) encoding,
per layer RX,RY,RZ on every qubit with angles drawn in the same order as the numpy code, then a CZ ring (q,q+1 mod n).
Features: <Z_i> (8) then <Z_i Z_j> for i<j (28) = 36. Run with .venv-qiskit."""
import itertools
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp


def angles(n, seed, layers):
    rng = np.random.default_rng(seed)
    return [[rng.uniform(0, 2 * np.pi) for q in range(n) for _ in range(3)] for _ in range(layers)]


def circuit(x, ang, n=8, entangle=True):
    qc = QuantumCircuit(n)
    for q in range(n): qc.ry(float(np.arctan(x[q])), q)
    for A in ang:
        k = 0
        for q in range(n):
            qc.rx(A[k], q); qc.ry(A[k + 1], q); qc.rz(A[k + 2], q); k += 3
        if entangle:
            for q in range(n): qc.cz(q, (q + 1) % n)
    return qc


def observables(n=8):
    lab = lambda qs: "".join("Z" if i in qs else "I" for i in range(n))[::-1]     # qiskit labels are little-endian
    return [SparsePauliOp(lab({i})) for i in range(n)] + [SparsePauliOp(lab({i, j})) for i, j in itertools.combinations(range(n), 2)]


def exact_features(X, seed, layers, n=8):
    ang, obs = angles(n, seed, layers), observables(n)
    return np.array([[float(np.real(Statevector(circuit(x, ang, n)).expectation_value(o))) for o in obs] for x in X])


if __name__ == "__main__":
    z = np.load("hw/exact_ref.npz"); X = z["X"]; worst = 0
    for s in (0, 3):
        for L in (1, 2, 3):
            d = np.abs(exact_features(X, s, L) - z[f"seed{s}_L{L}"]).max(); worst = max(worst, d); print(f"seed {s} layers {L}: max |qiskit - numpy| = {d:.2e}")
    print("PORT OK" if worst < 1e-9 else "PORT MISMATCH")
