"""Problem 2: single-norm and multi-norm (union) robustness evaluation.

Evaluates the Linf-trained, L2-trained and RAMP models on CIFAR-10 under
untargeted Linf PGD, untargeted L2 PGD, and the union threat model
  Delta = B_2(x, eps_2)  U  B_inf(x, eps_inf).

Usage:
  python run_eval.py                       # full 10k test set, k = 10
  python run_eval.py --k 20
  python run_eval.py --k 10 --random-start
  python run_eval.py --batches 5 --quiet   # quick smoke test
"""
import argparse
import json
import os
import time

import torch
from torchvision import datasets, transforms
from tqdm import tqdm

from resnet import PreActResNet18
from attacks import pgd_linf_untargeted, pgd_l2_untargeted

parser = argparse.ArgumentParser()
parser.add_argument('--k', type=int, default=10, help='number of PGD steps')
parser.add_argument('--eps-linf', type=float, default=8. / 255.)
parser.add_argument('--eps-l2', type=float, default=0.75)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--batches', type=int, default=0, help='0 = whole test set')
parser.add_argument('--random-start', action='store_true',
                    help='start PGD from a uniform random point in the ball')
parser.add_argument('--quiet', action='store_true', help='no progress bars')
parser.add_argument('--out', default='', help='write results json here')
parser.add_argument('--cpu', action='store_true')
args = parser.parse_args()

device = torch.device('cpu' if args.cpu else ('cuda' if torch.cuda.is_available() else 'cpu'))
torch.manual_seed(42)

test_dataset = datasets.CIFAR10('cifar10_data/', train=False, download=True,
                                transform=transforms.Compose([transforms.ToTensor()]))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

model = PreActResNet18(10, cuda=(device.type == 'cuda'), activation='softplus1').to(device)
model.eval()

CKPTS = {
    'Linf-trained': 'models/pretr_Linf.pth',
    'L2-trained':   'models/pretr_L2.pth',
    'RAMP':         'models/pretr_RAMP.pth',
}


def load(path):
    sd = torch.load(path, map_location=device, weights_only=False)
    for key in ('state_dict', 'model', 'net', 'model_state_dict'):
        if isinstance(sd, dict) and key in sd:
            sd = sd[key]
            break
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (len(missing), len(unexpected))
    model.eval()


@torch.no_grad()
def predict(x):
    return model(x).argmax(dim=1)


def evaluate(name):
    """One pass over the test set computing clean / Linf / L2 / union accuracy."""
    n = clean = rob_linf = rob_l2 = union = 0
    only_linf = only_l2 = both_broken = 0          # among clean-correct points
    n_clean_correct = 0
    max_linf = max_l2 = 0.0
    n_batches = args.batches if args.batches > 0 else len(test_loader)
    t0 = time.time()

    it = enumerate(test_loader)
    if not args.quiet:
        it = tqdm(it, total=n_batches, desc=f'{name:14s}')
    for i, (xb, yb) in it:
        if args.batches and i >= args.batches:
            break
        xb, yb = xb.to(device), yb.to(device)

        ok_clean = predict(xb) == yb

        x_linf = pgd_linf_untargeted(model, xb, yb, args.k, args.eps_linf,
                                     args.eps_linf / 4, random_start=args.random_start)
        x_l2 = pgd_l2_untargeted(model, xb, yb, args.k, args.eps_l2,
                                 args.eps_l2 / 4, random_start=args.random_start)

        ok_linf = predict(x_linf) == yb
        ok_l2 = predict(x_l2) == yb

        # union threat model: correct only if the model resists BOTH adversaries
        ok_union = ok_linf & ok_l2

        n += yb.size(0)
        clean += ok_clean.sum().item()
        rob_linf += ok_linf.sum().item()
        rob_l2 += ok_l2.sum().item()
        union += ok_union.sum().item()

        # which adversary breaks which points (restricted to clean-correct ones)
        n_clean_correct += ok_clean.sum().item()
        b_linf = ok_clean & ~ok_linf
        b_l2 = ok_clean & ~ok_l2
        only_linf += (b_linf & ~b_l2).sum().item()
        only_l2 += (b_l2 & ~b_linf).sum().item()
        both_broken += (b_linf & b_l2).sum().item()

        # sanity: verify the perturbations really are inside their balls
        max_linf = max(max_linf, (x_linf - xb).abs().max().item())
        max_l2 = max(max_l2, (x_l2 - xb).flatten(1).norm(dim=1).max().item())

    return dict(name=name, n=n, k=args.k, random_start=args.random_start,
                clean=100 * clean / n,
                linf=100 * rob_linf / n,
                l2=100 * rob_l2 / n,
                union=100 * union / n,
                only_linf=100 * only_linf / n,
                only_l2=100 * only_l2 / n,
                both_broken=100 * both_broken / n,
                n_clean_correct=n_clean_correct,
                max_linf=max_linf, max_l2=max_l2,
                secs=time.time() - t0)


print(f'device: {device} | torch {torch.__version__}')
print(f'PGD steps k = {args.k}, eps_inf = {args.eps_linf:.6f} (8/255), '
      f'eps_2 = {args.eps_l2}, eps_step = eps/4, '
      f"random_start = {args.random_start}\n")

rows = []
for name, path in CKPTS.items():
    load(path)
    r = evaluate(name)
    rows.append(r)
    print(f"{name:14s} standard {r['clean']:.2f} | Linf {r['linf']:.2f} | "
          f"L2 {r['l2']:.2f} | union {r['union']:.2f}   ({r['secs']:.0f}s)")
    print(f"               [check] max ||d||_inf = {r['max_linf']:.6f} "
          f"(<= {args.eps_linf:.6f}), max ||d||_2 = {r['max_l2']:.4f} (<= {args.eps_l2})")

w = 15
print()
print('=' * 74)
print(f"{'model':<{w}}{'standard':>11}{'Linf PGD':>11}{'L2 PGD':>11}{'union':>11}{'n':>9}")
print('-' * 74)
for r in rows:
    print(f"{r['name']:<{w}}{r['clean']:>10.2f}%{r['linf']:>10.2f}%"
          f"{r['l2']:>10.2f}%{r['union']:>10.2f}%{r['n']:>9}")
print('=' * 74)

print('\nBreakdown of clean-correct points, by which adversary succeeds '
      '(% of all test points):')
print(f"{'model':<{w}}{'only Linf':>12}{'only L2':>12}{'both':>12}"
      f"{'union gap':>12}")
print('-' * 74)
for r in rows:
    gap = min(r['linf'], r['l2']) - r['union']
    print(f"{r['name']:<{w}}{r['only_linf']:>11.2f}%{r['only_l2']:>11.2f}%"
          f"{r['both_broken']:>11.2f}%{gap:>11.2f}%")
print('-' * 74)
print('union gap = min(Linf acc, L2 acc) - union acc; > 0 means the two')
print('adversaries break genuinely different test points.')

if args.out:
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(rows, f, indent=2)
    print(f'\nwrote {args.out}')
