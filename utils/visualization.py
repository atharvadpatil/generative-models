"""Visualization utilities."""

import os
import torch
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple


def plot_loss(losses: List[float], title: str = "Training Loss", save_path: Optional[str] = None):
    """Plot loss curve."""
    plt.figure(figsize=(8, 5))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_three_panel(
    original: torch.Tensor,
    middle: torch.Tensor,
    generated: torch.Tensor,
    titles: Tuple[str, str, str],
    save_path: Optional[str] = None,
):
    """Create 3-panel comparison plot."""
    original = original.detach().cpu().numpy()
    middle = middle.detach().cpu().numpy()
    generated = generated.detach().cpu().numpy()
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    axs[0].scatter(original[:, 0], original[:, 1], s=2, alpha=0.5, color="steelblue")
    axs[0].set_xlim([-2, 2]); axs[0].set_ylim([-2, 2])
    axs[0].set_title(titles[0]); axs[0].set_aspect("equal"); axs[0].grid(True, alpha=0.3)
    
    axs[1].scatter(middle[:, 0], middle[:, 1], s=2, alpha=0.5, color="darkorange")
    axs[1].set_xlim([-4, 4]); axs[1].set_ylim([-4, 4])
    axs[1].set_title(titles[1]); axs[1].set_aspect("equal"); axs[1].grid(True, alpha=0.3)
    
    axs[2].scatter(generated[:, 0], generated[:, 1], s=2, alpha=0.5, color="seagreen")
    axs[2].set_xlim([-2, 2]); axs[2].set_ylim([-2, 2])
    axs[2].set_title(titles[2]); axs[2].set_aspect("equal"); axs[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
