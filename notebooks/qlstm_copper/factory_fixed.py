"""QLSTM with the circuit weights actually registered as trainable parameters.

Factory.QLSTM keeps its four TorchLayers in a plain dict (`self.VQC = {...}`), so nn.Module never
registers them: their weights receive gradients but are not in model.parameters(), the optimizer
never updates them, and they are missing from state_dict(). Every QLSTM result produced with
Factory.QShallowRegressionLSTM therefore used randomly initialised, untrained circuits with trained
classical encoder/decoder layers. This subclass changes only that (nn.ModuleDict); Factory.py is untouched.
The separate ry_params/RZ question in Factory.VQC is deliberately NOT changed here.
"""
from torch import nn

from Factory import QShallowRegressionLSTM


class _Gates(nn.Module):
    """Registers the gate circuits as submodules but keeps the self.VQC['forget'] access Factory.QLSTM.forward uses
    (a plain nn.ModuleDict cannot hold the key 'update': it clashes with ModuleDict.update)."""

    def __init__(self, gates):
        super().__init__()
        self.layers = nn.ModuleDict({f"g_{k}": v for k, v in gates.items()})

    def __getitem__(self, key):
        return self.layers[f"g_{key}"]


class QShallowRegressionLSTMTrained(QShallowRegressionLSTM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lstm.VQC = _Gates(self.lstm.VQC)
