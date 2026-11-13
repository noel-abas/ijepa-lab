"""Run the complete I-JEPA observation lab and save reproducible results."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform

import matplotlib
import numpy as np
from PIL import __version__ as pillow_version
import torch

from data_prep import DEFAULT_IMAGE, make_masks, plot_preparation, prepare_image
from loss_comparison import compare_losses, hand_examples, plot_loss_curves
from mask_comparison import compare_masks, plot_mask_comparison
from model import build_models, forward_pass, plot_architecture, plot_attention, shape_rows
from train_step import plot_momentum, run_training_step


PROJECT_DIR = Path(__file__).resolve().parent
UPSTREAM_REVISION = "52c1ae95d05f743e000e8f10a1f3a79b10cff048"


def write_table(path, rows):
    """Save a rectangular result table with a header, retaining full precision."""
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_lab(image_path=DEFAULT_IMAGE, output_dir=None, seed=7):
    """Generate six figures, numerical tables, and a run summary on CPU."""
    if output_dir is None:
        output_dir = PROJECT_DIR / "outputs"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    original, crop, image, crop_box = prepare_image(image_path)
    context_masks, target_masks = make_masks()
    plot_preparation(original, crop, context_masks, target_masks, output_dir / "01_data_preparation.png")

    models = build_models(seed=seed)
    with torch.no_grad():
        outputs = forward_pass(models, image, context_masks, target_masks)
    shapes = shape_rows(outputs)
    plot_architecture(outputs, output_dir / "02_architecture_flow.png")
    plot_attention(outputs, context_masks, target_masks, output_dir / "03_attention_weights.png")

    losses = compare_losses(outputs["predictions"], outputs["references"])
    examples = hand_examples()
    plot_loss_curves(output_dir / "04_loss_curves.png")

    training = run_training_step(models, image, context_masks, target_masks)
    plot_momentum(training["momentum_comparison"], output_dir / "05_target_update.png")

    mask_rows, mask_views = compare_masks(image, seed=seed)
    plot_mask_comparison(crop, mask_rows, mask_views, output_dir / "06_mask_comparison.png")
    write_table(output_dir / "tensor_shapes.csv", shapes)
    write_table(output_dir / "loss_examples.csv", examples)
    write_table(output_dir / "gradient_and_weight_changes.csv", training["components"])
    write_table(output_dir / "momentum_comparison.csv", training["momentum_comparison"])
    write_table(output_dir / "mask_comparison.csv", mask_rows)

    results = {
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "matplotlib": matplotlib.__version__,
            "pillow": pillow_version, "device": "cpu", "torch_threads": 1,
        },
        "setup": {
            "seed": seed,
            "upstream_revision": UPSTREAM_REVISION,
            "image_file": Path(image_path).name,
            "image_sha256": hashlib.sha256(Path(image_path).read_bytes()).hexdigest(),
            "original_size": list(original.size),
            "center_crop_box": list(crop_box),
            "image_shape": list(image.shape),
            "patch_size": 16,
            "context_indices": context_masks[0][0].tolist(),
            "target_indices": target_masks[0][0].tolist(),
            "initialization": "random; target initially copied from context",
            "encoder": "ViT-Tiny, width 192, 12 blocks, 3 heads",
            "predictor": "width 96, 2 blocks, 3 heads, output width 192",
        },
        "tensor_shapes": shapes,
        "baseline_losses": losses,
        "hand_examples": examples,
        "training_step": training,
        "mask_comparison": mask_rows,
    }
    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    results = run_lab(args.image, args.output_dir, args.seed)
    print("Completed one training step and two controlled comparisons.")
    print("Baseline Smooth L1:", results["baseline_losses"]["released_smooth_l1_mean"])
    for row in results["training_step"]["components"]:
        print(
            f"{row['component']}: {row['parameter_tensors_with_gradients']} parameter tensors with gradients; "
            f"tracked AdamW change {row['tracked_change_after_adamw_l2']:.8f}; "
            f"tracked change after EMA {row['tracked_change_after_ema_l2']:.8f}"
        )
    print("Results directory:", args.output_dir.resolve())


if __name__ == "__main__":
    main()
