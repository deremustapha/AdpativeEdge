import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

# ----- 3. Gradient Variance Estimation -----
def compute_gradient_variance(model, dataloader, loss_fn, num_batches=1, device='cuda'):
    grad_squares = defaultdict(list)
    grad_means = defaultdict(float)
    grad_counts = defaultdict(int)

    model.to(device)
    model.eval()

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
        inputs, targets = inputs.float().to(device), targets.long().to(device)
            
        model.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                g = param.grad.detach().clone()
                grad_squares[name].append(g ** 2)
                grad_means[name] += g
                grad_counts[name] += 1

    grad_variances = {}
    for name in grad_squares:
        mean_grad = grad_means[name] / grad_counts[name]
        squared_grads = torch.stack(grad_squares[name])
        var = squared_grads.mean(dim=0) - mean_grad ** 2
        grad_variances[name] = var

    return grad_variances

# ----- 4. Apply Sparse Mask -----
def apply_gradient_mask(model, grad_variances, sparsity=0.9):
    for name, param in model.named_parameters():
        if name in grad_variances:
            var = grad_variances[name]
            k = int((1 - sparsity) * var.numel())
            if k == 0:
                continue
            topk_vals, topk_idx = torch.topk(var.flatten(), k)
            mask = torch.zeros_like(var, dtype=torch.bool).flatten()
            mask[topk_idx] = True
            mask = mask.view_as(var)
            param.requires_grad = False
            param._grad_mask = mask

def mask_hook(grad, mask):
    return grad * mask

def register_gradient_hooks(model):
    for name, param in model.named_parameters():
        if hasattr(param, '_grad_mask'):
            param.requires_grad = True
            param.register_hook(lambda g, m=param._grad_mask: mask_hook(g, m))