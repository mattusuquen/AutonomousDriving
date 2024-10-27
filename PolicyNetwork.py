import torch
import torch.nn as nn

class PolicyNetwork(nn.Module):
    def __init__(self, n, dropout_rate=0.5, mean_range=1, stdev_coeff=0.1):
        super(PolicyNetwork, self).__init__()
        self.stdev_coeff = stdev_coeff
        self.mean_range = mean_range
        self.layers = nn.Sequential(
            nn.Linear(n, n),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(n, n // 2),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(n // 2, 2),
        )

    def forward(self, x):
        logits = self.layers(x)
        logits[:, 0] = torch.tanh(logits[:, 0]) * self.mean_range
        logits[:, 1] = torch.sigmoid(logits[:, 1]) * self.stdev_coeff
        
        return logits
