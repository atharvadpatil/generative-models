"""Multi-Layer Perceptron."""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Simple MLP with ReLU activations."""
    
    def __init__(self, input_dim: int, output_dim: int, n_hidden: int = 2, hidden_units: int = 64):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for _ in range(n_hidden):
            layers.append(nn.Linear(prev_dim, hidden_units))
            layers.append(nn.ReLU())
            prev_dim = hidden_units
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
