"""Measure how target size changes token counts and attention dimensions."""

import matplotlib.pyplot as plt
import torch

from data_prep import draw_patch_grid, make_masks
from model import build_models, forward_pass


def compare_masks(image, seed=7):
    """Run both masks on the same freshly initialized model, without updates."""
    models = build_models(seed=seed)
    rows, views = [], []
    for target_width in (2, 4):
        context_masks, target_masks = make_masks(target_height=2, target_width=target_width)
        with torch.no_grad():
            outputs = forward_pass(models, image, context_masks, target_masks)
        context_attention = outputs["attention"]["Context encoder"][0, 0]
        rows.append({
            "target_patches": target_masks[0].numel(),
            "context_patches": context_masks[0].numel(),
            "context_attention_shape": list(context_attention.shape),
            "context_query_key_pairs_per_head": context_attention.numel(),
            "target_attention_shape": list(outputs["attention"]["Target encoder"].shape[-2:]),
            "prediction_shape": list(outputs["predictions"].shape),
            "reference_shape": list(outputs["references"].shape),
        })
        views.append({"context_indices": context_masks[0], "attention": context_attention})
    return rows, views


def plot_mask_comparison(crop, rows, views, path):
    """Save context views above their measured first-head attention matrices."""
    figure, axes = plt.subplots(2, 2, figsize=(9, 8), layout="constrained")
    maximum = max(view["attention"].max().item() for view in views)
    for column, (row, view) in enumerate(zip(rows, views, strict=True)):
        draw_patch_grid(axes[0, column], crop, view["context_indices"])
        axes[0, column].set_title(
            f"{row['target_patches']} target patches, {row['context_patches']} context patches"
        )
        heatmap = axes[1, column].imshow(
            view["attention"].numpy(), cmap="viridis", vmin=0, vmax=maximum
        )
        ids = view["context_indices"][0].tolist()
        axes[1, column].set_xticks(range(len(ids)), ids, fontsize=8)
        axes[1, column].set_yticks(range(len(ids)), ids, fontsize=8)
        axes[1, column].set_xlabel("Visible key position")
        axes[1, column].set_ylabel("Visible query position")
        axes[1, column].set_title(
            f"Context attention {len(ids)} × {len(ids)}\n"
            f"{row['context_query_key_pairs_per_head']} query–key pairs per head"
        )
    figure.colorbar(heatmap, ax=axes[1, :], shrink=0.7, label="Attention weight")
    figure.suptitle("Same photograph and initial weights; different target sizes", fontsize=14)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
