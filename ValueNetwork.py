import torch
import torch.nn as nn

class ValueNetwork(nn.Module):
    def __init__(self, n):
        super(ValueNetwork, self).__init__()
        dropout_rate = 0.5

        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(n, n),
            nn.ReLU(),
            nn.LayerNorm(n), 
            nn.Dropout(dropout_rate),
            nn.Linear(n, n),
            nn.ReLU(),
            nn.Dropout(dropout_rate), 
            nn.Linear(n, 2),
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits
