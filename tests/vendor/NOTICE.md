# NOTICE — vendored third-party files

The three `.py` files in this directory are VERBATIM, unmodified copies of the
architecture files shipped with `nvidia/Nemotron-Labs-Diffusion-3B` on
HuggingFace (they are identical in the https://github.com/NVlabs/Nemotron-Labs-Diffusion
repository, source code licensed under Apache License 2.0):

| file | origin |
|---|---|
| `configuration_nemotron_labs_diffusion.py` | `nvidia/Nemotron-Labs-Diffusion-3B` (HF), same name |
| `modeling_ministral.py` | `nvidia/Nemotron-Labs-Diffusion-3B` (HF), same name |
| `modeling_nemotron_labs_diffusion.py` | `nvidia/Nemotron-Labs-Diffusion-3B` (HF), same name |

They are vendored ONLY so that `tests/make_tiny_repo.py` can build a tiny
random-weight instance of the exact same architecture for CPU smoke tests
without downloading anything. They are NOT used at runtime by `wet_ink.py` —
at runtime the helpers are imported from the module of the actually loaded
model (see `wet_ink.get_stock_helpers`).

Upstream license: Apache License 2.0 (source code), Copyright (c) 2026 NVIDIA
CORPORATION & AFFILIATES. Model weights are under the NVIDIA Open Model
License and are NOT included here.

Everything outside `tests/vendor/` in this repository is original code for
this project.
