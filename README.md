# CS 521 (Fall 2026) — Homework 1

Code for Homework 1. The written report (results, observations and the paper
review) is submitted separately as a PDF and is intentionally not tracked here.

Assignment starter code: [nalinwadhwa02/CCS521FA26HW](https://github.com/nalinwadhwa02/CCS521FA26HW)

## Problem 1 — FGSM attack

Starter: [`hw1/fgsm.py`](https://github.com/nalinwadhwa02/CCS521FA26HW/blob/main/hw1/fgsm.py)

| file | question |
|---|---|
| `fgsm.py` | **1.1** — targeted FGSM, target class `t = 0` |
| `fgsm_t1.py` | **1.2** — `t = 1`: one-step FGSM fails, targeted PGD succeeds within `ε = 0.5` |
| `fgsm_t1_minimal.py` | **1.2** — binary search for the minimal `L∞` radius (≈ 0.3012) |
| `explore_t1.py` | **1.2** — diagnostics: `ε` sweep, reachability of class 1, positive-homogeneity / cone analysis |

```bash
python fgsm.py
python fgsm_t1.py
python fgsm_t1_minimal.py
```

Runs on CPU-only PyTorch.

## Problem 2 — Single and multi-norm robustness with PGD

Starter: [`hw1/multi_norm_robustness.ipynb`](https://github.com/nalinwadhwa02/CCS521FA26HW/blob/main/hw1/multi_norm_robustness.ipynb)
· Pretrained weights: [Google Drive folder](https://drive.google.com/drive/folders/1y5E-OSGeS26R8rZAsgUyAbXPx99bXXa_?usp=sharing)

| file | question |
|---|---|
| `multi_norm_robustness.ipynb` | **2.1 + 2.2** — completed starter notebook, executed with outputs (`K = 10`) |
| `attacks.py` | **2.1** — `pgd_linf_untargeted`, `pgd_l2_untargeted` |
| `resnet.py` | `PreActResNet18`, verbatim from the starter |
| `run_eval.py` | **2.1 + 2.2** — standard / `L∞` / `L2` / union accuracy, plus which adversary breaks which points |
| `plot_results.py` | builds `results/robustness.png` |
| `results/*.json` | raw numbers for `k` = 10, 20, 50 and the random-start run |

The notebook is self-contained (the ResNet and both attacks are inlined as cells).
`attacks.py` / `resnet.py` / `run_eval.py` exist to run the larger `k` = 20 and 50
sweeps outside the notebook; `run_eval.py` imports the other two.

```bash
# needs a CUDA GPU and models/pretr_{Linf,L2,RAMP}.pth
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
gdown --folder "https://drive.google.com/drive/folders/1y5E-OSGeS26R8rZAsgUyAbXPx99bXXa_" -O models

python run_eval.py --k 10 --out results/k10.json
python run_eval.py --k 50 --out results/k50.json
python plot_results.py
```

`build_notebook.py` regenerates the notebook from `attacks.py` and `resnet.py` so
the module code and the notebook cells cannot drift apart.

## Problem 3 — Paper review

Review of [Testing Robustness Against Unforeseen Adversaries](https://arxiv.org/abs/1908.08016)
(Kaufmann et al., ImageNet-UA / UA2 version). Prose only — no code; see the
submitted PDF.

## Setup

```bash
# Problem 1 only (CPU is fine)
pip install torch

# Problem 2 (needs CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install numpy matplotlib tqdm gdown
```

`cifar10_data/` and `models/` are gitignored: the dataset is downloaded
automatically by torchvision, and the weights come from the Drive link above.
