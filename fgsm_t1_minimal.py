"""Part 2: bracket the minimal L-inf radius for a targeted attack to class 1.

Heavy multi-restart PGD at a ladder of fixed radii, then a binary search.
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

x0 = torch.rand((1, 10))
t = 1
tgt = torch.tensor([t], dtype=torch.long)


def margin_batch(v):
    lg = N(v)
    other = torch.cat([lg[:, :t], lg[:, t + 1:]], dim=1)
    return lg[:, t] - other.max(dim=1).values


def feasible(radius, restarts=4096, steps=800, gen=None):
    """Batched targeted PGD in the L-inf ball of the given radius."""
    lo, hi = x0 - radius, x0 + radius
    d = (torch.rand((restarts, 10), generator=gen) * 2 - 1) * radius
    d[0] = 0
    v = (x0 + d).clamp(lo, hi).requires_grad_()
    alpha = radius / 40.0
    best = None
    for _ in range(steps):
        if v.grad is not None:
            v.grad.zero_()
        margin_batch(v).sum().backward()
        with torch.no_grad():
            v += alpha * v.grad.sign()      # ascend the target margin
            v.clamp_(lo, hi)
            m = margin_batch(v)
            if (m > 0).any():
                ok = v[m > 0]
                norms = (ok - x0).abs().amax(dim=1)
                i = norms.argmin()
                cand = ok[i:i + 1].clone()
                if best is None or (cand - x0).abs().max() < (best - x0).abs().max():
                    best = cand
    return best


print("=== radius ladder (4096 restarts each) ===")
results = {}
for r in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
    a = feasible(r)
    results[r] = a
    if a is None:
        print(f"  radius {r:4.2f} -> fail")
    else:
        print(f"  radius {r:4.2f} -> SUCCESS, Linf = {(a-x0).abs().max().item():.5f}")

ok_radii = [r for r, a in results.items() if a is not None]
fail_radii = [r for r, a in results.items() if a is None]
print()

if ok_radii:
    hi = min(ok_radii)
    lo = max(fail_radii) if fail_radii else 0.0
    best = results[hi]
    print(f"=== binary search between {lo} and {hi} ===")
    for _ in range(16):
        mid = 0.5 * (lo + hi)
        a = feasible(mid, restarts=4096, steps=800)
        if a is not None:
            hi, best = mid, a
            print(f"  {mid:.6f} SUCCESS")
        else:
            lo = mid
            print(f"  {mid:.6f} fail")
    print(f"\nminimal L-inf radius bracketed to [{lo:.6f}, {hi:.6f}]")

    d = best - x0
    print()
    print("================ MINIMAL-NORM RESULT, t = 1 ================")
    print("x                 :", [round(z, 5) for z in x0.squeeze(0).tolist()])
    print("adv_x             :", [round(z, 5) for z in best.squeeze(0).tolist()])
    print("delta             :", [round(z, 5) for z in d.squeeze(0).tolist()])
    print("new class         :", N(best).argmax(dim=1).item())
    print("logits(adv_x)     :", [round(z, 6) for z in N(best).detach().squeeze(0).tolist()])
    print("||x - adv_x||_inf :", d.abs().max().item())
    print("||x - adv_x||_2   :", d.norm(p=2).item())
    print("||x - adv_x||_1   :", d.abs().sum().item())
