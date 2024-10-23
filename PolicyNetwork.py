import torch
import torch.nn as nn

class PolicyNetwork(nn.Module):
    def __init__(self, n, dropout_rate=0.5):
        super(PolicyNetwork, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(n, n * 2),
            nn.Tanh(),
            nn.LayerNorm(n * 2), 
            nn.Linear(n * 2, n * 2),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(n * 2, n),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(n, 2),
        )

    def forward(self, x):
        logits = self.layers(x)
        logits[:,1] = nn.Sigmoid()(logits[:,1])
        return logits
