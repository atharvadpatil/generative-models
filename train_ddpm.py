#!/usr/bin/env python3
"""Train DDPM diffusion model."""

import argparse
import os
import random
import numpy as np
import torch
import yaml

from models import DDPM
from utils import load_data, create_dataloader, plot_loss, plot_three_panel


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config(path: str) -> dict:
    """Load config from YAML file."""
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train DDPM")
    parser.add_argument("--config", "-c", type=str, default="configs/ddpm.yaml", help="Path to config file")
    # Override options
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=689)
    parser.add_argument("--data", type=str, default="689_data.csv")
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    # Load config from YAML
    cfg = load_config(args.config)
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    
    # Allow CLI args to override config
    epochs = args.epochs or train_cfg.get("epochs", 1000)
    batch_size = args.batch_size or train_cfg.get("batch_size", 4096)
    lr = args.lr or train_cfg.get("learning_rate", 1e-3)
    lr_decay = train_cfg.get("lr_decay", 0.999)
    
    # Load data
    print(f"Loading data from: {args.data}")
    data = load_data(args.data)
    n_samples, dim = data.shape
    print(f"Dataset: {n_samples} samples, {dim}D")
    
    loader = create_dataloader(data, batch_size=batch_size)
    
    # Create model
    model = DDPM(
        x_dim=dim,
        diffusion_steps=model_cfg.get("diffusion_steps", 2500),
        beta2_start=model_cfg.get("beta2_start", 1e-4),
        beta2_end=model_cfg.get("beta2_end", 2e-2),
        time_embed_dim=model_cfg.get("time_embed_dim", 32),
        hidden_units=model_cfg.get("hidden_units", 64),
    )
    
    print(f"\nDDPM: {model_cfg.get('diffusion_steps', 2500)} diffusion steps")
    print(f"Config: {args.config}")
    
    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)
    
    losses = []
    print(f"\nTraining for {epochs} epochs...")
    print("-" * 40)
    
    for epoch in range(1, epochs + 1):
        batch_losses = []
        for (x_batch,) in loader:
            optimizer.zero_grad()
            loss = model.loss(x_batch)
            loss.backward()
            optimizer.step()
            scheduler.step()
            batch_losses.append(loss.item())
        
        avg_loss = sum(batch_losses) / len(batch_losses)
        losses.append(avg_loss)
        
        if epoch % 50 == 0 or epoch == epochs:
            print(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.6f}")
    
    print("-" * 40)
    print(f"Final loss: {losses[-1]:.6f}")
    
    # Save checkpoint
    os.makedirs("outputs/checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "outputs/checkpoints/ddpm.pt")
    print("Saved: outputs/checkpoints/ddpm.pt")
    
    # Visualize
    model.eval()
    with torch.no_grad():
        x_L = model.forward_diffuse(data)
        print("Generating samples (this may take a while)...")
        x_gen = model.sample(n_samples)
    
    plot_loss(losses, title="DDPM Training Loss", save_path="outputs/figures/ddpm_loss.png")
    print("Saved: outputs/figures/ddpm_loss.png")
    
    plot_three_panel(
        data, x_L, x_gen,
        titles=("Original Data", "Forward Diffused", "Generated Samples"),
        save_path="outputs/figures/ddpm_results.png",
    )
    print("Saved: outputs/figures/ddpm_results.png")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
