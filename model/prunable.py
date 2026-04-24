import torch
import torch.nn as nn

class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.gate_scores = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, x):
        gates = torch.sigmoid(10 * self.gate_scores)
        pruned_weights = self.weight * gates
        return x @ pruned_weights.t() + self.bias

    def get_gates(self):
        return torch.sigmoid(10 * self.gate_scores)