"""RealNVP Normalizing Flow."""

import math
import torch
import torch.nn as nn
from typing import Tuple, Optional

from .mlp import MLP


class CouplingLayer(nn.Module):
    """Affine coupling layer for RealNVP."""
    
    def __init__(self, dim: int, n_hidden: int = 2, hidden_units: int = 128, flip: bool = False):
        super().__init__()
        self.dim = dim
        self.half = dim // 2
        self.flip = flip
        self.scale_alpha = 0.9
        
        self.net = MLP(
            input_dim=self.half,
            output_dim=2 * self.half,
            n_hidden=n_hidden,
            hidden_units=hidden_units,
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward: x -> z, returns (z, log_det)."""
        single = (x.dim() == 1)
        if single:
            x = x.unsqueeze(0)
        
        # Split
        left, right = x[..., :self.half], x[..., self.half:]
        xa, xb = (right, left) if self.flip else (left, right)
        
        # Affine transform
        h = self.net(xa)
        s, t = torch.chunk(h, 2, dim=-1)
        s = self.scale_alpha * torch.tanh(s)
        
        zb = xb * torch.exp(s) + t
        
        # Merge
        if self.flip:
            z = torch.cat([zb, xa], dim=-1)
        else:
            z = torch.cat([xa, zb], dim=-1)
        
        log_det = s.sum(dim=-1)
        
        if single:
            return z.squeeze(0), log_det.squeeze(0)
        return z, log_det
    
    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Inverse: z -> x."""
        single = (z.dim() == 1)
        if single:
            z = z.unsqueeze(0)
        
        left, right = z[..., :self.half], z[..., self.half:]
        za, zb = (right, left) if self.flip else (left, right)
        
        h = self.net(za)
        s, t = torch.chunk(h, 2, dim=-1)
        s = self.scale_alpha * torch.tanh(s)
        
        xb = (zb - t) * torch.exp(-s)
        
        if self.flip:
            x = torch.cat([xb, za], dim=-1)
        else:
            x = torch.cat([za, xb], dim=-1)
        
        if single:
            return x.squeeze(0)
        return x


class RealNVP(nn.Module):
    """RealNVP Normalizing Flow model."""
    
    def __init__(self, dim: int, n_coupling_layers: int = 14, n_hidden: int = 2, hidden_units: int = 128):
        super().__init__()
        self.dim = dim
        
        self.layers = nn.ModuleList([
            CouplingLayer(dim=dim, n_hidden=n_hidden, hidden_units=hidden_units, flip=bool(i % 2))
            for i in range(n_coupling_layers)
        ])
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward: x -> z, returns (z, log_det)."""
        single = (x.dim() == 1)
        if single:
            x = x.unsqueeze(0)
        
        z = x
        log_det_total = torch.zeros(x.shape[0], device=x.device)
        
        for layer in self.layers:
            z, log_det = layer(z)
            log_det_total = log_det_total + log_det
        
        if single:
            return z.squeeze(0), log_det_total.squeeze(0)
        return z, log_det_total
    
    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Inverse: z -> x."""
        single = (z.dim() == 1)
        if single:
            z = z.unsqueeze(0)
        
        x = z
        for layer in reversed(self.layers):
            x = layer.inverse(x)
        
        if single:
            return x.squeeze(0)
        return x
    
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Log probability under the model."""
        z, log_det = self.forward(x)
        log_pz = -0.5 * (z ** 2).sum(dim=-1) - 0.5 * self.dim * math.log(2 * math.pi)
        return log_pz + log_det
    
    def sample(self, n_samples: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """Generate samples."""
        if device is None:
            device = next(self.parameters()).device
        z = torch.randn(n_samples, self.dim, device=device)
        return self.inverse(z)
