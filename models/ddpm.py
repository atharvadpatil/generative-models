"""Denoising Diffusion Probabilistic Model (DDPM)."""

import math
import torch
import torch.nn as nn
from typing import Tuple, Optional


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding."""
    
    def __init__(self, max_steps: int, embed_dim: int = 32):
        super().__init__()
        self.max_steps = max_steps
        self.embed_dim = embed_dim
        
        half = embed_dim // 2
        freqs = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), half))
        self.register_buffer("freqs", freqs)
    
    def forward(self, t_indices: torch.Tensor) -> torch.Tensor:
        t = t_indices.float().unsqueeze(1) / float(self.max_steps)
        ang = t * self.freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)
        
        if emb.shape[1] < self.embed_dim:
            padding = torch.zeros(emb.size(0), self.embed_dim - emb.size(1), device=emb.device)
            emb = torch.cat([emb, padding], dim=1)
        return emb


class MuNet(nn.Module):
    """Network for predicting posterior mean."""
    
    def __init__(self, x_dim: int = 2, time_dim: int = 32, hidden_units: int = 64, max_steps: int = 2500):
        super().__init__()
        self.time_emb = TimeEmbedding(max_steps=max_steps, embed_dim=time_dim)
        self.net = nn.Sequential(
            nn.Linear(x_dim + time_dim, hidden_units),
            nn.SiLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.SiLU(),
            nn.Linear(hidden_units, x_dim),
        )
    
    def forward(self, x: torch.Tensor, t_indices: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_emb(t_indices)
        return self.net(torch.cat([x, t_emb], dim=1))


class DDPM(nn.Module):
    """Denoising Diffusion Probabilistic Model."""
    
    def __init__(
        self,
        x_dim: int = 2,
        diffusion_steps: int = 2500,
        beta2_start: float = 1e-4,
        beta2_end: float = 2e-2,
        time_embed_dim: int = 32,
        hidden_units: int = 64,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.L = diffusion_steps
        self.eps = 1e-12
        
        # Noise schedule
        beta2 = torch.linspace(beta2_start, beta2_end, diffusion_steps)
        alpha2 = 1.0 - beta2
        alpha = torch.sqrt(alpha2)
        alpha_bar = torch.cumprod(alpha, dim=0)
        beta = torch.sqrt(beta2)
        
        self.register_buffer("beta2", beta2)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("beta", beta)
        
        self.mu_net = MuNet(x_dim=x_dim, time_dim=time_embed_dim, hidden_units=hidden_units, max_steps=diffusion_steps)
    
    def loss(self, x0: torch.Tensor) -> torch.Tensor:
        """Compute training loss."""
        batch_size = x0.size(0)
        device = x0.device
        
        t = torch.randint(low=1, high=self.L - 1, size=(batch_size,), device=device)
        
        # Sample x_t
        ab = self.alpha_bar[t]
        mean = ab.unsqueeze(1) * x0
        std = torch.sqrt(1.0 - ab ** 2).unsqueeze(1)
        x_t = mean + std * torch.randn_like(x0)
        
        # Posterior params
        a = self.alpha[t]
        abar_prev = torch.where(t > 0, self.alpha_bar[t - 1], torch.ones_like(a))
        
        num_mu = (a * (1 - abar_prev ** 2)).unsqueeze(1) * x_t + (abar_prev * (1 - a ** 2)).unsqueeze(1) * x0
        den = (1 - (a ** 2) * (abar_prev ** 2)).unsqueeze(1).clamp_min(self.eps)
        mu_tilde = num_mu / den
        
        sigma2_tilde = ((1 - a ** 2) * (1 - abar_prev ** 2) / den.squeeze(1)).unsqueeze(1).clamp_min(self.eps)
        
        mu_pred = self.mu_net(x_t, t)
        w = 1.0 / (2.0 * sigma2_tilde)
        return (w * (mu_pred - mu_tilde) ** 2).sum(dim=1).mean()
    
    @torch.no_grad()
    def sample(self, n_samples: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """Generate samples via reverse diffusion."""
        if device is None:
            device = next(self.parameters()).device
        
        x = torch.randn(n_samples, self.x_dim, device=device)
        
        for i in reversed(range(self.L)):
            t_idx = torch.full((n_samples,), i, dtype=torch.long, device=device)
            mu = self.mu_net(x, t_idx)
            
            a = self.alpha[i]
            abar_prev = 1.0 if i == 0 else self.alpha_bar[i - 1]
            sigma2 = max(((1 - a ** 2) * (1 - abar_prev ** 2) / (1 - (a ** 2) * (abar_prev ** 2))).item(), self.eps)
            
            if i > 0:
                x = mu + math.sqrt(sigma2) * torch.randn_like(x)
            else:
                x = mu
        
        return x
    
    @torch.no_grad()
    def forward_diffuse(self, x0: torch.Tensor) -> torch.Tensor:
        """Full forward diffusion to x_L."""
        ab_L = self.alpha_bar[-1]
        return ab_L * x0 + torch.sqrt(1.0 - ab_L ** 2) * torch.randn_like(x0)
