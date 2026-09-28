"""CS 521 HW1, Problem 1, Part 2: targeted attack with target class t = 1.

Part A demonstrates that single-step FGSM cannot reach class 1 within the
allowed budget (or anywhere near it).
Part B succeeds using targeted PGD (iterative FGSM with random restarts) and
reports the norms of the perturbation.
"""
import torch
import torch.nn as nn

torch.manual_seed(13)

N = nn.Sequential(nn.Linear(10, 10, bias=False),
                  nn.ReLU(),
                  nn.Linear(10, 10, bias=False),
                  nn.ReLU(),
                  nn.Linear(10, 3, bias=False))
for p in N.parameters():
    p.requires_grad_(False)

x = torch.rand((1, 10))
x.requires_grad_()

t = 1
epsReal = 0.5
eps = epsReal - 1e-7

L = nn.CrossEntropyLoss()
tgt = torch.tensor([t], dtype=torch.long)

original_class = N(x).argmax(dim=1).item()
print("Original Class:", original_class)
print("logits(x)     :", [round(z, 5) for z in N(x).detach().squeeze(0).tolist()])
assert original_class == 2
print()

# ===================== Part A: plain targeted FGSM fails =====================
L(N(x), tgt).backward()
grad_sign = x.grad.sign()
adv_fgsm = (x - eps * grad_sign).detach()

print("=== A) single-step targeted FGSM, eps = 0.5 ===")
print("class :", N(adv_fgsm).argmax(dim=1).item(), "(target was", t, ")  -> FAILS")
print("logits:", [round(z, 5) for z in N(adv_fgsm).detach().squeeze(0).tolist()])
print("Linf  :", (adv_fgsm - x.detach()).abs().max().item())

hits = [i * 0.01 for i in range(1, 2001)
        if N((x - i * 0.01 * grad_sign).detach()).argmax(dim=1).item() == t]
if hits:
    print(f"step sizes along -sign(grad) reaching class 1: "
          f"[{min(hits):.2f}, {max(hits):.2f}]  (all >> 0.5)")
else:
    print("no step size along -sign(grad) reaches class 1")
print()

# ================== Part B: targeted PGD / iterative FGSM ====================
x0 = x.detach().clone()


def target_margin(v):
    """> 0 exactly when v is classified as the target class t."""
    lg = N(v)
    other = torch.cat([lg[:, :t], lg[:, t + 1:]], dim=1)
    return lg[:, t] - other.max(dim=1).values


def targeted_pgd(radius, restarts=4096, steps=800, seed=0):
    """Iterative FGSM with random restarts, projected onto the L-inf ball."""
    g = torch.Generator().manual_seed(seed)
    lo, hi = x0 - radius, x0 + radius
    delta = (torch.rand((restarts, 10), generator=g) * 2 - 1) * radius
    delta[0] = 0                                    # one plain (unperturbed) start
    v = (x0 + delta).clamp(lo, hi).requires_grad_()
    alpha = radius / 40.0
    best = None
    for _ in range(steps):
        if v.grad is not None:
            v.grad.zero_()
        target_margin(v).sum().backward()           # ascend the target margin
        with torch.no_grad():
            v += alpha * v.grad.sign()              # signed-gradient step
            v.clamp_(lo, hi)                        # project back into the ball
            m = target_margin(v)
            if (m > 0).any():
                ok = v[m > 0]
                i = (ok - x0).abs().amax(dim=1).argmin()
                cand = ok[i:i + 1].clone()
                if best is None or (cand - x0).abs().max() < (best - x0).abs().max():
                    best = cand
    return best


print("=== B) targeted PGD inside the eps = 0.5 ball ===")
adv_x = targeted_pgd(eps)
assert adv_x is not None, "PGD failed"

new_class = N(adv_x).argmax(dim=1).item()
d = adv_x - x0

print("New Class:", new_class)
print("logits   :", [round(z, 6) for z in N(adv_x).detach().squeeze(0).tolist()])
print()
print("x     :", [round(z, 5) for z in x0.squeeze(0).tolist()])
print("adv_x :", [round(z, 5) for z in adv_x.squeeze(0).tolist()])
print("delta :", [round(z, 5) for z in d.squeeze(0).tolist()])
print()
print("||x - adv_x||_inf =", d.abs().max().item())
print("||x - adv_x||_2   =", d.norm(p=2).item())
print("||x - adv_x||_1   =", d.abs().sum().item())

assert new_class == t
assert torch.norm(x0 - adv_x, p=float('inf')) <= epsReal
print("\nSUCCESS: classified as t = 1 and within the L-inf budget of 0.5")
