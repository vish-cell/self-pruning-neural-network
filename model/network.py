import torch.nn as nn
from model.prunable import PrunableLinear

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = PrunableLinear(3072, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        return self.fc3(x)

    def sparsity_loss(self):
        loss = 0
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                g = m.get_gates()
                loss += g.mean()
        return loss

    def sparsity_level(self, threshold=0.05):
        total, zero = 0, 0
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                g = m.get_gates()
                total += g.numel()
                zero += (g < threshold).sum().item()
        return 100 * zero / total
