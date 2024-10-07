import torch
import torch.nn as nn

class PolicyNetwork(nn.Module):
    def __init__(self, n):
        super().__init__()
        dropout_rate = 0.5

        self.flatten = nn.Flatten()
        self.policy_layers = nn.Sequential(
            nn.Linear(n, n * 2),
            nn.ReLU(),
            nn.LayerNorm(n * 2), 
            nn.Linear(n * 2, n * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(n * 2, n),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(n, 2),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.policy_layers(x)
        return logits
