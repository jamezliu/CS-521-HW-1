"""Untargeted Linf and L2 PGD attacks (Problem 2.1)."""
import torch

DELTA = 1e-10   # guards the divide-by-zero in the L2 gradient normalisation


def pgd_linf_untargeted(model, x, labels, k, eps, eps_step, random_start=False):
    """Untargeted Linf PGD: k steps of FGSM, each projected back onto B_inf(x, eps).

    Projection onto the Linf ball is just a coordinate-wise clamp of the
    perturbation to [-eps, eps]; we additionally clamp to the image domain [0,1].
    """
    model.eval()
    ce_loss = torch.nn.CrossEntropyLoss()
    x = x.detach()
    adv_x = x.clone().detach()
    if random_start:
        adv_x = adv_x + torch.empty_like(adv_x).uniform_(-eps, eps)
        adv_x = torch.clamp(adv_x, 0.0, 1.0).detach()
    for _ in range(k):
        adv_x.requires_grad_(True)
        model.zero_grad()
        output = model(adv_x)
        loss = ce_loss(output, labels)
        loss.backward()
        with torch.no_grad():
            # untargeted => ASCEND the loss of the true label
            adv_x = adv_x + eps_step * adv_x.grad.sign()
            # project onto the eps-ball around the ORIGINAL x, then onto [0,1]
            delta = torch.clamp(adv_x - x, min=-eps, max=eps)
            adv_x = torch.clamp(x + delta, 0.0, 1.0)
        adv_x = adv_x.detach()
    return adv_x


def pgd_l2_untargeted(model, x, labels, k, eps, eps_step, random_start=False):
    """Untargeted L2 PGD.

    Step:        x^(t+1) = x^(t) + eta * g^(t) / (||g^(t)||_2 + delta)
    Projection:  x_proj  = x + eps * (x^(t+1) - x) / max(||x^(t+1) - x||_2, eps)

    Both the gradient normalisation and the projection are computed per example
    in the batch, so the flattened norms are taken over the C*H*W dimensions.
    """
    model.eval()
    ce_loss = torch.nn.CrossEntropyLoss()
    x = x.detach()
    adv_x = x.clone().detach()
    batch_size = x.size()[0]
    if random_start:
        d = torch.randn_like(adv_x)
        d = d / (d.view(batch_size, -1).norm(p=2, dim=1).view(-1, 1, 1, 1) + DELTA)
        r = torch.rand(batch_size, 1, 1, 1, device=x.device) ** (1.0 / d[0].numel())
        adv_x = torch.clamp(adv_x + eps * r * d, 0.0, 1.0).detach()
    for _ in range(k):
        adv_x.requires_grad_(True)
        model.zero_grad()
        output = model(adv_x)
        loss = ce_loss(output, labels)
        loss.backward()
        grad = adv_x.grad.data
        with torch.no_grad():
            # normalise the gradient to unit L2 norm per example
            grad_norms = grad.view(batch_size, -1).norm(p=2, dim=1) + DELTA
            grad = grad / grad_norms.view(batch_size, 1, 1, 1)
            adv_x = adv_x + eps_step * grad

            # project the perturbation back onto the L2 ball of radius eps
            delta = adv_x - x
            delta_norms = delta.view(batch_size, -1).norm(p=2, dim=1)
            factor = eps / torch.clamp(delta_norms, min=eps)   # == eps / max(||delta||, eps)
            delta = delta * factor.view(batch_size, 1, 1, 1)

            adv_x = torch.clamp(x + delta, 0.0, 1.0)
        adv_x = adv_x.detach()
    return adv_x
