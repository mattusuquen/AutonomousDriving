import torch
import torch.nn as nn

class ValueNetwork(nn.Module):
    def __init__(self, n, dropout_rate=0.5):
        super(ValueNetwork, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(n, n),
            nn.Tanh(),
            nn.LayerNorm(n), 
            nn.Dropout(dropout_rate),
            nn.Linear(n, n),
            nn.Tanh(),
            nn.Dropout(dropout_rate), 
            nn.Linear(n, 1),
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits
