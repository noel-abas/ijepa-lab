"""Observe one AdamW step and the target encoder's moving-average update."""

from itertools import chain

import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F

from model import forward_pass


def update_target(context_encoder, target_encoder, momentum):
    """Update every target parameter in place without building a gradient graph."""
    if not 0 <= momentum <= 1:
        raise ValueError("Momentum must be between zero and one.")
    with torch.no_grad():
        for context_parameter, target_parameter in zip(
            context_encoder.parameters(), target_encoder.parameters(), strict=True
        ):
            target_parameter.mul_(momentum).add_(context_parameter, alpha=1.0 - momentum)


def momentum_comparison(old_target, updated_context, momenta=(0.90, 0.99, 0.999)):
    """Use the same two endpoint tensors for every candidate update."""
    if old_target.shape != updated_context.shape:
        raise ValueError("Target and context weights must have identical shapes.")
    gap = (updated_context - old_target).norm().item()
    if gap == 0:
        raise ValueError("A nonzero context update is needed to measure a movement fraction.")
    rows = []
    for momentum in momenta:
        if not 0 <= momentum <= 1:
            raise ValueError("Momentum must be between zero and one.")
        candidate = momentum * old_target + (1.0 - momentum) * updated_context
        movement = (candidate - old_target).norm().item()
        rows.append({
            "momentum": momentum,
            "initial_gap_l2": gap,
            "target_movement_l2": movement,
            "measured_gap_fraction": movement / gap,
            "expected_gap_fraction": 1.0 - momentum,
        })
    return rows


def run_training_step(models, image, context_masks, target_masks, momentum=0.99):
    """Perform exactly one forward/backward/AdamW/EMA cycle on CPU.

    The fixed learning rate and one parameter group keep this observation small.
    The full upstream recipe uses schedules and separate weight-decay groups.
    All movement measurements below concern the explicitly named weight tensor.
    """
    if not 0 <= momentum <= 1:
        raise ValueError("Momentum must be between zero and one.")
    context_encoder, target_encoder, predictor = models
    components = {
        "Context encoder": context_encoder,
        "Predictor": predictor,
        "Target encoder": target_encoder,
    }
    tracked = {
        "Context encoder": ("patch_embed.proj.weight", context_encoder.patch_embed.proj.weight),
        "Predictor": ("predictor_proj.weight", predictor.predictor_proj.weight),
        "Target encoder": ("patch_embed.proj.weight", target_encoder.patch_embed.proj.weight),
    }
    before = {name: parameter.detach().clone() for name, (_, parameter) in tracked.items()}
    optimizer = torch.optim.AdamW(
        chain(context_encoder.parameters(), predictor.parameters()),
        lr=1e-3, weight_decay=0.04,
    )
    optimizer.zero_grad(set_to_none=True)
    outputs = forward_pass(models, image, context_masks, target_masks)
    loss = F.smooth_l1_loss(outputs["predictions"], outputs["references"], beta=1.0)
    if not torch.isfinite(loss):
        raise RuntimeError("The loss must be finite before updating parameters.")
    loss.backward()

    rows = []
    for name, component in components.items():
        gradients = [parameter.grad for parameter in component.parameters() if parameter.grad is not None]
        gradient_norm = 0.0
        if gradients:
            gradient_norm = torch.stack([gradient.detach().norm() for gradient in gradients]).norm().item()
        rows.append({
            "component": name,
            "tracked_parameter": tracked[name][0],
            "parameter_tensors_with_gradients": len(gradients),
            "gradient_l2_all_parameters": gradient_norm,
        })

    optimizer.step()
    updated_context = tracked["Context encoder"][1].detach().clone()
    comparison = momentum_comparison(before["Target encoder"], updated_context)
    for row in rows:
        parameter = tracked[row["component"]][1]
        row["tracked_change_after_adamw_l2"] = (parameter.detach() - before[row["component"]]).norm().item()
    update_target(context_encoder, target_encoder, momentum)
    for row in rows:
        parameter = tracked[row["component"]][1]
        row["tracked_change_after_ema_l2"] = (parameter.detach() - before[row["component"]]).norm().item()

    expected = momentum * before["Target encoder"] + (1.0 - momentum) * updated_context
    torch.testing.assert_close(tracked["Target encoder"][1], expected)
    if any(parameter.grad is not None for parameter in target_encoder.parameters()):
        raise RuntimeError("The frozen target encoder must have empty gradient fields.")
    return {
        "smooth_l1_before_update": loss.item(),
        "learning_rate": 1e-3,
        "weight_decay": 0.04,
        "applied_momentum": momentum,
        "components": rows,
        "momentum_comparison": comparison,
    }


def plot_momentum(rows, path):
    """Compare measured movements with the same-start prediction from 1 − m."""
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
    labels = [str(row["momentum"]) for row in rows]
    axes[0].bar(labels, [row["target_movement_l2"] for row in rows], color="#315b9c", width=0.55)
    axes[0].set_ylabel("L2 change of tracked target kernel")
    axes[0].set_title("Same starting weights for each candidate")
    axes[1].bar(labels, [row["measured_gap_fraction"] for row in rows], color="#315b9c", width=0.55, label="Measured")
    axes[1].plot(labels, [row["expected_gap_fraction"] for row in rows], "o", color="#bd5a2b", label="Expected 1 − m")
    axes[1].set_ylabel("Fraction of initial weight gap")
    axes[1].set_title("Measured movement follows 1 − m")
    axes[1].legend()
    for ax in axes:
        ax.set_xlabel("Momentum m")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    figure.suptitle("Target encoder moving-average comparison", fontsize=15)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
