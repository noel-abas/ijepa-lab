# Sources and attribution

## Original architecture and implementation

Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent,
Michael Rabbat, Yann LeCun, and Nicolas Ballas. *Self-Supervised Learning from
Images with a Joint-Embedding Predictive Architecture*. CVPR 2023.

- [Research paper](https://arxiv.org/abs/2301.08243)
- [Original repository](https://github.com/facebookresearch/ijepa)
- Pinned commit `52c1ae95d05f743e000e8f10a1f3a79b10cff048`.
- Copyright © Meta Platforms, Inc. and affiliates.
- The vendored source retains its original CC BY-NC 4.0 license and copyright
  headers. See [vendor/ijepa/LICENSE](vendor/ijepa/LICENSE).
- The Python files under `vendor/ijepa/src`, the upstream README, and the
  ViT-H/14 configuration were copied without source changes from that commit.

| Concept | Original file to inspect |
|---|---|
| Patch projection and row-major tokens | `vendor/ijepa/src/models/vision_transformer.py`, `PatchEmbed` |
| Fixed sine/cosine positional embeddings | Same file, `get_2d_sincos_pos_embed` |
| Q, K, V and scaled attention | Same file, `Attention.forward` |
| Residual connections and feedforward network | Same file, `Block.forward` |
| Context masking before Transformer blocks | Same file, `VisionTransformer.forward` |
| Predictor mask tokens and output projection | Same file, `VisionTransformerPredictor.forward` |
| Target normalization, gathering, Smooth L1, EMA | `vendor/ijepa/src/train.py`, nested training-step functions |
| Full optimizer groups | `vendor/ijepa/src/helper.py`, `init_opt` |
| Random crop and pixel normalization | `vendor/ijepa/src/transforms.py`, `make_transforms` |
| Production multi-block sampling | `vendor/ijepa/src/masks/multiblock.py`, `MaskCollator` |
| ViT-H/14 ImageNet recipe | `vendor/ijepa/configs/in1k_vith14_ep300.yaml` |

Our modules adapt the forward, loss, and update sequence for a CPU demonstration.
They additionally measure shapes, inspect gradients, save figures, and perform
controlled comparisons. The notebook and report prompts are a study aid.

## Photograph

`assets/astronaut.png` depicts astronaut Eileen Collins. NASA photograph,
distributed as scikit-image's `astronaut` example. The scikit-image documentation
identifies its NASA source and public-domain status.

- [Dataset description and credit](https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.astronaut)
- [Original NASA image record linked by scikit-image](https://flic.kr/p/r9qvLn)
- [Included PNG source](https://raw.githubusercontent.com/scikit-image/scikit-image/v0.19.3/skimage/data/astronaut.png)

The bundled PNG is unmodified. The lab creates a crop, resized view, and masked
visualizations as part of its figures. It uses this photograph as an inspection
input. The original paper pretrains on ImageNet.

## PyTorch behavior

- [Smooth L1 loss](https://docs.pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html)
- [AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html)
- [Autograd introduction](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)

Dependencies retain their respective licenses. The bundled photograph and
upstream I-JEPA source have the attributions stated above.
