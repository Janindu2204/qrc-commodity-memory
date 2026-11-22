"""Exact features from the project's own simulator for 20 fixed random inputs (reference for the Qiskit port). Run with the main venv from qlstm_copper/."""
import numpy as np, hourly_databento_qrc as qrc
rng = np.random.default_rng(7); X = rng.normal(size=(20, 8))
F = {f"seed{s}_L{L}": qrc.reservoir_features(X, 8, s, True, L) for s in (0, 3) for L in (1, 2, 3)}
np.savez("hw/exact_ref.npz", X=X, **F); print({k: v.shape for k, v in F.items()})
