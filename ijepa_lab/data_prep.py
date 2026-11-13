"""Prepare one photograph and explicit patch indices for the I-JEPA lab."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
import torch


DEFAULT_IMAGE = Path(__file__).resolve().parent / "assets" / "astronaut.png"
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


def prepare_image(image_path=DEFAULT_IMAGE, image_size=64, crop_fraction=0.8):
    """Return the RGB original, resized center crop, normalized tensor, and crop box.

    The fixed center crop makes repeated comparisons reproducible. Meta's
    ImageNet recipe instead samples random resized crops during training.
    """
    if image_size <= 0 or not 0 < crop_fraction <= 1:
        raise ValueError("Use a positive image size and a crop fraction in (0, 1].")
    with Image.open(image_path) as source:
        original = ImageOps.exif_transpose(source).convert("RGB")
    width, height = original.size
    side = max(1, round(min(width, height) * crop_fraction))
    left, top = (width - side) // 2, (height - side) // 2
    crop_box = (left, top, left + side, top + side)
    crop = original.crop(crop_box).resize(
        (image_size, image_size), Image.Resampling.BICUBIC
    )
    pixels = np.asarray(crop, dtype=np.float32).copy() / 255.0
    image = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0)
    mean = torch.tensor(IMAGE_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGE_STD).view(1, 3, 1, 1)
    return original, crop, (image - mean) / std, crop_box


def make_masks(grid_side=4, target_height=2, target_width=2):
    """Return one centered rectangular target and its complementary context.

    Each index tensor has shape (1, number_of_selected_patches). The list around
    each tensor is the representation expected by Meta's implementation.
    """
    if min(grid_side, target_height, target_width) <= 0:
        raise ValueError("Grid and target dimensions must be positive.")
    if max(target_height, target_width) > grid_side:
        raise ValueError("The target rectangle must fit inside the patch grid.")
    if target_height * target_width == grid_side ** 2:
        raise ValueError("At least one context patch must remain visible.")
    grid = torch.arange(grid_side ** 2).reshape(grid_side, grid_side)
    top = (grid_side - target_height) // 2
    left = (grid_side - target_width) // 2
    target = grid[top:top + target_height, left:left + target_width].flatten()
    visible = torch.ones(grid_side ** 2, dtype=torch.bool)
    visible[target] = False
    context = grid.flatten()[visible]
    return [context.unsqueeze(0)], [target.unsqueeze(0)]


def masked_view(crop, selected_indices, patch_size=16):
    """Create a display image with unselected positions dimmed.

    Dimming is only a visualization. The context Transformer receives a gathered
    sequence of visible tokens, rather than this dimmed image.
    """
    pixels = np.asarray(crop, dtype=np.float32) / 255.0
    height, width = pixels.shape[:2]
    if height != width or height % patch_size:
        raise ValueError("Use a square crop divisible by the patch size.")
    grid_side = height // patch_size
    visible = np.zeros((grid_side, grid_side), dtype=bool)
    visible.flat[selected_indices.reshape(-1).tolist()] = True
    visible_pixels = visible.repeat(patch_size, axis=0).repeat(patch_size, axis=1)
    return np.where(visible_pixels[..., None], pixels, pixels * 0.12)


def draw_patch_grid(ax, crop, selected_indices=None, patch_size=16):
    """Draw row-major indices over either the full crop or a selected view."""
    pixels = crop
    if selected_indices is not None:
        pixels = masked_view(crop, selected_indices, patch_size)
    ax.imshow(pixels, interpolation="nearest")
    side = crop.width // patch_size
    for boundary in range(0, crop.width + 1, patch_size):
        ax.axhline(boundary - 0.5, color="white", linewidth=0.6, alpha=0.7)
        ax.axvline(boundary - 0.5, color="white", linewidth=0.6, alpha=0.7)
    for index in range(side ** 2):
        row, col = divmod(index, side)
        ax.text(
            (col + 0.5) * patch_size - 0.5,
            (row + 0.5) * patch_size - 0.5,
            str(index), ha="center", va="center", color="white", fontsize=10,
            bbox={"facecolor": "black", "alpha": 0.65, "pad": 1.5, "edgecolor": "none"},
        )
    ax.set_axis_off()


def plot_preparation(original, crop, context_masks, target_masks, path):
    """Save the original, crop grid, and the two complementary patch selections."""
    figure, axes = plt.subplots(1, 4, figsize=(12, 3.7), layout="constrained")
    axes[0].imshow(original)
    axes[0].set_axis_off()
    axes[0].set_title(f"Original photograph\n{original.width} × {original.height}")
    draw_patch_grid(axes[1], crop)
    axes[1].set_title("Center crop resized to 64 × 64\n16 patches, each 16 × 16")
    draw_patch_grid(axes[2], crop, context_masks[0])
    axes[2].set_title(f"Visible context\n{context_masks[0].numel()} selected patches")
    draw_patch_grid(axes[3], crop, target_masks[0])
    axes[3].set_title(f"Target positions\n{target_masks[0].numel()} selected patches")
    figure.suptitle("One image, shared patch positions", fontsize=15)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
