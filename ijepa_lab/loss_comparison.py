"""Compare the paper's squared L2 objective with the release's Smooth L1."""

import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F


def compare_losses(predictions, references):
    """Report both original reductions and a common per-feature reduction.

    The sum represents the paper's expression for this lab's single image and
    single target block. It does not implement batching over multiple blocks.
    """
    if predictions.shape != references.shape:
        raise ValueError("Predictions and references must have identical shapes.")
    if predictions.numel() == 0:
        raise ValueError("At least one feature is needed to calculate a loss.")
    with torch.no_grad():
        errors = predictions - references
        squared_sum = errors.square().sum().item()
        squared_mean = errors.square().mean().item()
        smooth_mean = F.smooth_l1_loss(predictions, references, beta=1.0).item()
    return {
        "feature_count": predictions.numel(),
        "paper_squared_l2_sum": squared_sum,
        "squared_error_mean": squared_mean,
        "released_smooth_l1_mean": smooth_mean,
    }


def hand_examples():
    """Check the essay's two-coordinate calculations with PyTorch."""
    rows = []
    for second_error in (-2.0, -4.0):
        errors = torch.tensor([0.5, second_error])
        rows.append({
            "first_residual": 0.5,
            "second_residual": second_error,
            "squared_l2_sum": errors.square().sum().item(),
            "squared_error_mean": errors.square().mean().item(),
            "smooth_l1_mean": F.smooth_l1_loss(errors, torch.zeros_like(errors)).item(),
        })
    return rows


def plot_loss_curves(path):
    """Save per-feature penalties and their derivatives with respect to residuals."""
    errors = torch.linspace(-4, 4, 401, requires_grad=True)
    squared = errors.square()
    smooth = F.smooth_l1_loss(errors, torch.zeros_like(errors), beta=1.0, reduction="none")
    squared_gradient = torch.autograd.grad(squared.sum(), errors)[0]
    smooth_gradient = torch.autograd.grad(smooth.sum(), errors)[0]
    x = errors.detach().numpy()
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
    axes[0].plot(x, squared.detach().numpy(), label="Squared error", color="#315b9c")
    axes[0].plot(x, smooth.detach().numpy(), label="Smooth L1, beta = 1", color="#bd5a2b")
    axes[0].set_ylabel("Penalty for one feature")
    axes[0].set_title("Loss contribution")
    axes[1].plot(x, squared_gradient.numpy(), label="Squared error", color="#315b9c")
    axes[1].plot(x, smooth_gradient.numpy(), label="Smooth L1, beta = 1", color="#bd5a2b")
    axes[1].set_ylabel("Derivative with respect to residual")
    axes[1].set_title("Gradient before averaging")
    for ax in axes:
        ax.axvline(-1, color="gray", linestyle=":", linewidth=1)
        ax.axvline(1, color="gray", linestyle=":", linewidth=1)
        ax.axhline(0, color="gray", linewidth=0.6)
        ax.set_xlabel("Residual = prediction − reference")
        ax.grid(alpha=0.2)
        ax.legend()
    figure.suptitle("Same residual, two penalty rules", fontsize=15)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
