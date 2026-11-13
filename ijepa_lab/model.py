"""Instantiate and inspect Meta's original I-JEPA encoders and predictor."""

from copy import deepcopy
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import torch
from torch.nn import functional as F


# Keep the vendored source byte-for-byte identical to the attributed upstream.
UPSTREAM = Path(__file__).resolve().parent / "vendor" / "ijepa"
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))

from src.masks.utils import apply_masks
from src.models.vision_transformer import vit_predictor, vit_tiny


def build_models(seed=7, image_size=64, patch_size=16):
    """Create a seeded ViT-Tiny context, an identical target, and a small predictor."""
    if image_size <= 0 or patch_size <= 0 or image_size % patch_size:
        raise ValueError("The image size must be a positive multiple of the patch size.")
    torch.manual_seed(seed)
    context_encoder = vit_tiny(img_size=[image_size], patch_size=patch_size)
    target_encoder = deepcopy(context_encoder).requires_grad_(False)
    predictor = vit_predictor(
        num_patches=(image_size // patch_size) ** 2,
        embed_dim=context_encoder.embed_dim,
        predictor_embed_dim=96,
        depth=2,
        num_heads=context_encoder.num_heads,
    )
    return context_encoder, target_encoder, predictor


def forward_pass(models, image, context_masks, target_masks):
    """Trace one context mask and one target mask through the two branches.

    Target normalization and gathering follow src/train.py in the original
    repository. Attention hooks observe the first layer's post-softmax weights.
    """
    if len(context_masks) != 1 or len(target_masks) != 1 or image.shape[0] != 1:
        raise ValueError("This lab traces one image, one context mask, and one target mask.")
    context_encoder, target_encoder, predictor = models
    attention = {}

    def capture_attention(name):
        def hook(_module, _inputs, outputs):
            attention[name] = outputs[1].detach().cpu()
        return hook

    layers = {
        "Context encoder": context_encoder.blocks[0].attn,
        "Target encoder": target_encoder.blocks[0].attn,
        "Predictor": predictor.predictor_blocks[0].attn,
    }
    handles = [
        layer.register_forward_hook(capture_attention(name))
        for name, layer in layers.items()
    ]
    try:
        with torch.no_grad():
            patch_tokens = context_encoder.patch_embed(image)
            target_features = target_encoder(image)
            normalized_targets = F.layer_norm(
                target_features, (target_features.shape[-1],)
            )
            references = apply_masks(normalized_targets, target_masks)
        context_features = context_encoder(image, context_masks)
        predictions = predictor(context_features, context_masks, target_masks)
    finally:
        for handle in handles:
            handle.remove()

    if predictions.shape != references.shape:
        raise RuntimeError("Prediction and reference shapes must agree.")
    return {
        "patch_tokens": patch_tokens,
        "context_features": context_features,
        "target_features": target_features,
        "references": references,
        "predictions": predictions,
        "attention": attention,
    }


def shape_rows(outputs):
    """Return JSON-friendly feature and attention dimensions."""
    rows = [
        {"component": name, "shape": list(outputs[name].shape)}
        for name in ("patch_tokens", "context_features", "target_features", "references", "predictions")
    ]
    rows.extend(
        {"component": name + " attention", "shape": list(weights.shape)}
        for name, weights in outputs["attention"].items()
    )
    return rows


def plot_attention(outputs, context_masks, target_masks, path):
    """Save one head's actual first-layer weights with patch positions labeled."""
    figure, axes = plt.subplots(1, 3, figsize=(12, 4.5), layout="constrained")
    names = ("Context encoder", "Target encoder", "Predictor")
    context_ids = context_masks[0][0].tolist()
    target_ids = target_masks[0][0].tolist()
    labels = (
        [str(index) for index in context_ids],
        [str(index) for index in range(outputs["target_features"].shape[1])],
        [f"C{index}" for index in context_ids] + [f"T{index}" for index in target_ids],
    )
    maximum = max(outputs["attention"][name][0, 0].max().item() for name in names)
    for ax, name, ticks in zip(axes, names, labels, strict=True):
        weights = outputs["attention"][name][0, 0].numpy()
        heatmap = ax.imshow(weights, cmap="viridis", vmin=0, vmax=maximum)
        ax.set_title(f"{name}\n{weights.shape[0]} × {weights.shape[1]}")
        ax.set_xticks(range(len(ticks)), ticks, rotation=90, fontsize=7)
        ax.set_yticks(range(len(ticks)), ticks, fontsize=7)
        ax.set_xlabel("Key position")
        ax.set_ylabel("Query position")
    figure.colorbar(heatmap, ax=axes, shrink=0.7, label="Attention weight after softmax")
    figure.suptitle("First layer, first head, before training", fontsize=15)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_architecture(outputs, path):
    """Draw shapes measured in the baseline forward pass."""
    shapes = {key: tuple(outputs[key].shape) for key in (
        "patch_tokens", "context_features", "target_features", "predictions", "references"
    )}
    figure, ax = plt.subplots(figsize=(9, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_axis_off()
    positions = {
        "image": (5, 9), "context": (2.5, 6.8), "target": (7.5, 6.8),
        "predictor": (2.5, 4.3), "gather": (7.5, 4.3), "loss": (5, 1.7),
    }
    labels = {
        "image": f"Shared image crop, aligned positions\nEach branch embeds patches {shapes['patch_tokens']}",
        "context": f"Context encoder\nVisible tokens only\n{shapes['context_features']}",
        "target": f"Target encoder\nAll crop tokens\n{shapes['target_features']}",
        "predictor": f"Predictor + positional mask tokens\nPredictions {shapes['predictions']}",
        "gather": f"Normalize, then gather target positions\nReferences {shapes['references']}",
        "loss": "Smooth L1 mean\nCompare matching positions and features",
    }
    for start, end in (
        ("image", "context"), ("image", "target"), ("context", "predictor"),
        ("target", "gather"), ("predictor", "loss"), ("gather", "loss"),
    ):
        x1, y1 = positions[start]
        x2, y2 = positions[end]
        ax.add_patch(FancyArrowPatch(
            (x1, y1 - 0.70), (x2, y2 + 0.70),
            arrowstyle="-|>", mutation_scale=16, color="#46576d", linewidth=1.5,
        ))
    for name, (x, y) in positions.items():
        ax.add_patch(FancyBboxPatch(
            (x - 2.15, y - 0.55), 4.3, 1.1,
            boxstyle="round,pad=0.12", facecolor="#edf3fb", edgecolor="#315984",
        ))
        ax.text(x, y, labels[name], ha="center", va="center", fontsize=10)
    ax.set_title("Observed data flow for one target block", fontsize=15, pad=16)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
