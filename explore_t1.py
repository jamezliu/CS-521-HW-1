"""Diagnostics for Part 2: why does single-step targeted FGSM fail for t = 1?"""
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
t = 1
L = nn.CrossEntropyLoss()
tgt = torch.tensor([t], dtype=torch.long)


def logits(v):
    return N(v).detach().squeeze(0)


print("logits(x)        =", logits(x).tolist())
print("original class   =", logits(x).argmax().item())
print()

# ---- 1. plain targeted FGSM at the required budget -------------------------
def fgsm(eps):
    xv = x.clone().requires_grad_()
    L(N(xv), tgt).backward()
    return (xv - eps * xv.grad.sign()).detach()


print("--- single-step targeted FGSM, sweeping eps ---")
for eps in [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]:
    a = fgsm(eps)
    lg = logits(a)
    print(f"eps={eps:7.2f}  class={lg.argmax().item()}  logits={[round(z,4) for z in lg.tolist()]}")
print()

# ---- 2. is class 1 reachable at all? unconstrained maximization of logit_1 --
print("--- unconstrained: maximize (logit_1 - max other logit) ---")
v = x.clone().requires_grad_()
opt = torch.optim.Adam([v], lr=0.05)
for i in range(4000):
    opt.zero_grad()
    lg = N(v).squeeze(0)
    margin = lg[1] - torch.max(lg[0], lg[2])
    (-margin).backward()
    opt.step()
lg = logits(v)
print("best margin       =", (lg[1] - torch.max(lg[0], lg[2])).item())
print("class             =", lg.argmax().item())
print("logits            =", [round(z, 4) for z in lg.tolist()])
print("Linf from x       =", (v - x).abs().max().item())
print()

# ---- 3. how large can logit_1 get anywhere in input space? ------------------
print("--- max logit_1 over random restarts (unbounded) ---")
best = -1e9
for r in range(20):
    v = (x + torch.randn_like(x) * (r + 1)).requires_grad_()
    opt = torch.optim.Adam([v], lr=0.1)
    for i in range(2000):
        opt.zero_grad()
        lg = N(v).squeeze(0)
        (-(lg[1] - torch.max(lg[0], lg[2]))).backward()
        opt.step()
    lg = logits(v)
    m = (lg[1] - torch.max(lg[0], lg[2])).item()
    if m > best:
        best, bestlg, bestv = m, lg, v.detach().clone()
print("best margin over restarts =", best)
print("logits at best            =", [round(z, 4) for z in bestlg.tolist()])
print("argmax                    =", bestlg.argmax().item())
print()

# ---- 4. structure of the last layer ---------------------------------------
W3 = N[4].weight.detach()
print("--- final layer weights (rows = classes) ---")
print("row0:", [round(z, 3) for z in W3[0].tolist()])
print("row1:", [round(z, 3) for z in W3[1].tolist()])
print("row2:", [round(z, 3) for z in W3[2].tolist()])
print()
print("row1 - row0:", [round(z, 3) for z in (W3[1] - W3[0]).tolist()])
print("row1 - row2:", [round(z, 3) for z in (W3[1] - W3[2]).tolist()])
print()
print("Note: post-ReLU activations feeding the final layer are always >= 0.")
h = x
for layer in list(N)[:4]:
    h = layer(h)
print("activations into final layer for x:", [round(z, 4) for z in h.detach().squeeze(0).tolist()])
print()

# ---- 5. no biases => positive homogeneity => decision regions are cones -----
print("--- positive homogeneity check (the net has no bias terms) ---")
for c in [0.1, 1.0, 10.0]:
    lg = logits(c * x.detach())
    print(f"  N({c:4.1f}*x) = {[round(z,5) for z in lg.tolist()]}   class={lg.argmax().item()}")
print("  N(c*v) = c*N(v) for c > 0, so every decision region is a cone through 0")
print("  and only the *direction* of the input determines the predicted class.")
print()

print("--- how much of direction space does each class own? ---")
counts = {0: 0, 1: 0, 2: 0}
trials = 200000
for _ in range(trials):
    counts[logits(torch.randn(1, 10)).argmax().item()] += 1
for k in sorted(counts):
    print(f"  class {k}: {counts[k]:6d}  ({100*counts[k]/trials:6.3f}%)")
