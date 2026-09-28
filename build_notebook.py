"""Emit the completed multi_norm_robustness.ipynb from the starter template."""
import json

import resnet  # noqa: F401  (just to assert the module is importable)

with open('resnet.py') as f:
    resnet_src = f.read()

# strip the module docstring + imports so the notebook cell is self-contained
resnet_body = resnet_src.split('\n', 1)[1].lstrip('\n')
resnet_body = resnet_body.replace('import torch\nimport torch.nn as nn\n'
                                  'import torch.nn.functional as F\n\n\n', '')

with open('attacks.py') as f:
    attacks_src = f.read()
attacks_body = attacks_src.split('"""', 2)[2].lstrip('\n')
attacks_body = attacks_body.replace('import torch\n\n', '')


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(True)}


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.splitlines(True)}


cells = []

cells.append(md("""# Set up for dataset and model

Package installation, loading, and dataloaders. There's also a resnet18 model defined."""))

cells.append(code("""import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

from torchvision import datasets, transforms

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
batch_size = 64

np.random.seed(42)
torch.manual_seed(42)


## Dataloaders
test_dataset = datasets.CIFAR10('cifar10_data/', train=False, download=True, transform=transforms.Compose(
    [transforms.ToTensor()]
))

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)"""))

cells.append(code(resnet_body + """

# intialize the model
model = PreActResNet18(10, cuda=True, activation='softplus1').to(device)
model.eval()"""))

cells.append(md("""# Implement the Attacks

Both attacks are untargeted, so each step *ascends* the cross-entropy loss of the
**true** label, then projects back onto the eps-ball around the original `x` and
onto the image domain `[0, 1]`.

* **Linf**: the step is `eps_step * sign(grad)` and the projection is a coordinate-wise
  clamp of the perturbation to `[-eps, eps]`.
* **L2**: the gradient is normalised to unit L2 norm per example,
  `g / (||g||_2 + 1e-10)`, and the projection rescales the perturbation by
  `eps / max(||delta||_2, eps)` (a no-op when already inside the ball)."""))

cells.append(code(attacks_body.rstrip()))

cells.append(md("""# Evaluate Single and Multi-Norm Robust Accuracy

In this section, we evaluate the model on the Linf and L2 attacks as well as union accuracy."""))

cells.append(code('''K = 10   # number of PGD steps


def load_model(path):
    sd = torch.load(path, map_location=device, weights_only=False)
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing and not unexpected
    model.eval()
    return model


def test_model_on_single_attack(model, attack='pgd_linf', eps=0.1, k=K):
    model.eval()
    tot_test, tot_acc = 0.0, 0.0
    for batch_idx, (x_batch, y_batch) in tqdm(enumerate(test_loader), total=len(test_loader), desc="Evaluating"):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        if attack == 'pgd_linf':
            x_adv = pgd_linf_untargeted(model, x_batch, y_batch, k, eps, eps / 4)
        elif attack == 'pgd_l2':
            x_adv = pgd_l2_untargeted(model, x_batch, y_batch, k, eps, eps / 4)
        else:
            x_adv = x_batch          # no attack -> standard accuracy

        with torch.no_grad():
            pred = torch.max(model(x_adv), dim=1)[1]
        tot_acc += (pred == y_batch).sum().item()
        tot_test += x_batch.size(0)

    acc = tot_acc / tot_test
    label = 'Standard accuracy' if attack not in ('pgd_linf', 'pgd_l2') else 'Robust accuracy'
    print('%s %.5lf' % (label, acc), f'on {attack} attack with eps = {eps}')
    return acc'''))

cells.append(md("""## Standard (clean) accuracy"""))

cells.append(code("""for name in ['Linf', 'L2', 'RAMP']:
    load_model(f'models/pretr_{name}.pth')
    print(f'=== {name} ===')
    test_model_on_single_attack(model, attack='none')"""))

cells.append(md("""## Single-Norm Robust Accuracy"""))

cells.append(code("""# Evaluate on Linf attack with different models with eps = 8/255
model.load_state_dict(torch.load('models/pretr_Linf.pth'))
model.eval()
test_model_on_single_attack(model, attack='pgd_linf', eps=8. / 255.)

model.load_state_dict(torch.load('models/pretr_L2.pth'))
model.eval()
test_model_on_single_attack(model, attack='pgd_linf', eps=8. / 255.)

model.load_state_dict(torch.load('models/pretr_RAMP.pth'))
model.eval()
test_model_on_single_attack(model, attack='pgd_linf', eps=8. / 255.)"""))

cells.append(code("""# Evaluate on L2 attack with different models with eps = 0.75
model.load_state_dict(torch.load('models/pretr_Linf.pth'))
model.eval()
test_model_on_single_attack(model, attack='pgd_l2', eps=0.75)

model.load_state_dict(torch.load('models/pretr_L2.pth'))
model.eval()
test_model_on_single_attack(model, attack='pgd_l2', eps=0.75)

model.load_state_dict(torch.load('models/pretr_RAMP.pth'))
model.eval()
test_model_on_single_attack(model, attack='pgd_l2', eps=0.75)"""))

cells.append(md("""## Multi-Norm Robust Accuracy

Union accuracy over `Delta = B_2(x, eps_2) U B_inf(x, eps_inf)`: a point counts as
correct only if the model classifies it correctly under **both** adversaries. A single
point that either attack breaks is counted as a failure."""))

cells.append(code('''def test_model_on_multi_attacks(model, eps_linf=8. / 255., eps_l2=0.75, k=K):
    model.eval()
    tot_test, tot_acc = 0.0, 0.0
    for batch_idx, (x_batch, y_batch) in tqdm(enumerate(test_loader), total=len(test_loader), desc="Evaluating"):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        x_adv_linf = pgd_linf_untargeted(model, x_batch, y_batch, k, eps_linf, eps_linf / 4)
        x_adv_l2 = pgd_l2_untargeted(model, x_batch, y_batch, k, eps_l2, eps_l2 / 4)

        ## calculate union accuracy: correct only if both attacks are correct

        with torch.no_grad():
            out = model(x_adv_linf)
            pred_linf = torch.max(out, dim=1)[1]
            out = model(x_adv_l2)
            pred_l2 = torch.max(out, dim=1)[1]

        correct = (pred_linf == y_batch) & (pred_l2 == y_batch)
        tot_acc += correct.sum().item()
        tot_test += x_batch.size(0)

    print('Robust accuracy %.5lf' % (tot_acc / tot_test), f'on multi attacks')
    return tot_acc / tot_test'''))

cells.append(code("""# Evaluate on multi-norm attacks with different models with eps_linf = 8./255, eps_l2 = 0.75
model.load_state_dict(torch.load('models/pretr_Linf.pth'))
model.eval()
test_model_on_multi_attacks(model)

model.load_state_dict(torch.load('models/pretr_L2.pth'))
model.eval()
test_model_on_multi_attacks(model)

model.load_state_dict(torch.load('models/pretr_RAMP.pth'))
model.eval()
test_model_on_multi_attacks(model)"""))

cells.append(md("""## Results

See `WRITEUP.md` for the full tables, the convergence check over the number of PGD
steps, and the discussion of the accuracy differences between the three models."""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.6"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

with open('multi_norm_robustness.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print(f'wrote multi_norm_robustness.ipynb with {len(cells)} cells')
