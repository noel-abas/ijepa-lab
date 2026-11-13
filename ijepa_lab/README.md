# I-JEPA observation lab

Investigate how one image moves through the context encoder, target encoder,
predictor, loss, and parameter updates. The project uses the original I-JEPA
implementation with a small model and a single photograph. The notebook connects
each experiment to questions for a personalized architecture report.

## Run the project

Use Python 3.11 or 3.12. From this folder, create an environment and install the
dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_lab.py
```

Open `ijepa_lab.ipynb` in VS Code or Jupyter and select that environment's Python
kernel for the guided walkthrough. Run the cells from the top. The notebook
contains saved outputs from a completed CPU run. In Colab, upload and extract the
whole project folder, change into it, and then run the notebook or command.

The photograph and required upstream code are included. After installing the
dependencies, running the experiments requires no downloads.

To use your own photograph, supply its path. The loader handles RGB conversion,
EXIF orientation, a fixed center crop, resizing, and ImageNet normalization.

```bash
python run_lab.py --image /path/to/photo.jpg --output-dir outputs_my_photo
```

## Experimental setup

| Choice | Value |
|---|---|
| Input | One NASA photograph of astronaut Eileen Collins |
| Crop | Center square covering 80% of the shorter side, resized to 64 × 64 |
| Normalization | ImageNet mean `(0.485, 0.456, 0.406)` and std `(0.229, 0.224, 0.225)` |
| Patch size | 16 × 16 pixels, giving 16 positions |
| Baseline target | Positions `[5, 6, 9, 10]` |
| Baseline context | The other 12 positions |
| Encoders | Original `vit_tiny`, width 192, 12 blocks, 3 attention heads |
| Predictor | Original `vit_predictor`, width 96, 2 blocks, 3 heads |
| Initialization | Seed 7, random weights, target copied from context |
| Update | One Smooth L1 backward pass, AdamW step, then EMA at momentum 0.99 |
| Device | CPU, one PyTorch thread |

The released ViT-H/14 ImageNet configuration uses random resized crops, larger
models, four sampled target blocks, parameter groups, and training schedules.
Our fixed input, one target block, single optimizer group, and fixed learning
rate make the mechanics inspectable. The predictor returns embeddings at the
requested patch positions. Its outputs have shape `(1, 4, 192)` in the baseline.

The experiments measure tensor flow, penalties, gradients, and parameter changes.
The paper's experiments provide the evidence for learned representation quality.
The attention figures show a randomly initialized model before training.

## Files to study

| File | What it implements |
|---|---|
| `data_prep.py` | Photograph preparation, patch indices, context/target selections, image views |
| `model.py` | Construction of the three components, forward trace, attention hooks, architecture figure |
| `loss_comparison.py` | Squared L2 and Smooth L1 calculations, controlled residuals, penalty and gradient plots |
| `train_step.py` | Backpropagation, AdamW, gradient observations, same-start EMA comparisons |
| `mask_comparison.py` | Four-target versus eight-target comparison with fixed image and model weights |
| `run_lab.py` | One entry point that produces all figures and numerical results |
| `ijepa_lab.ipynb` | Guided execution, code inspection, saved outputs, reflection questions |
| `vendor/ijepa/src/models/vision_transformer.py` | Original patch embedding, attention, Transformer blocks, encoders, and predictor |

`model.py` imports Meta's implementation from `vendor/ijepa`. That folder also
contains the original training loop, mask sampler, transforms, training
configuration, and license for comparison. Its Python files are unmodified.
The project-specific files are the educational experiment around that code.
See [SOURCES.md](SOURCES.md) for the pinned revision and attribution.

## Figures and tables

| Output | Suggested report use |
|---|---|
| `outputs/01_data_preparation.png` | Explain the crop, grid, and visible versus requested positions |
| `outputs/02_architecture_flow.png` | Follow the observed shapes through both branches |
| `outputs/03_attention_weights.png` | Identify query/key axes and the effect of sequence length |
| `outputs/04_loss_curves.png` | Explain penalty slopes using a common per-feature basis |
| `outputs/05_target_update.png` | Relate measured movement to `1 − momentum` |
| `outputs/06_mask_comparison.png` | Compare visible tokens and context attention dimensions |
| `outputs/tensor_shapes.csv` | Exact feature and attention shapes |
| `outputs/loss_examples.csv` | Checked two-coordinate calculations |
| `outputs/gradient_and_weight_changes.csv` | Gradient counts and changes in named weight tensors |
| `outputs/momentum_comparison.csv` | Same-start candidate target updates |
| `outputs/mask_comparison.csv` | Token counts, query/key pairs, and prediction/reference shapes |
| `outputs/results.json` | Full run configuration, environment, and numerical results |

The paper-style squared L2 sum and the release's Smooth L1 mean use different
reductions. A mean squared error is also reported so the reduction can be held
constant when comparing penalty rules. The loss figure plots individual feature
penalties and derivatives before averaging.

Each EMA candidate starts from the same saved target kernel and the same
post-AdamW context kernel. The live target is updated once at momentum 0.99.
Weight-movement tables identify the particular kernel or projection measured.
The mask comparison uses the same fresh models for both masks with no updates.

Use the notebook's questions to develop your own observations. For each report
section, connect the concept and equation to a code location, a measured output,
and your interpretation. Preserve source attribution when uploading this folder
to your GitHub repository.
