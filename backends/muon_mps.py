"""
MuonAdamW optimizer adapted for PyTorch MPS backend.
Based on the original autoresearch train.py optimizer with fixes for MPS compatibility:
- Removed @torch.compile decorators (not supported on MPS)
- Scalar tensors moved to device before arithmetic
- CUDA-specific calls removed
"""

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Polar Express coefficients (Newton-Schulz orthogonalization)
# ---------------------------------------------------------------------------

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

# ---------------------------------------------------------------------------
# Step functions (no torch.compile — MPS does not support it)
# ---------------------------------------------------------------------------

def adamw_step(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    """AdamW update step, MPS-compatible (no torch.compile).
    All arithmetic done in float32 to avoid MPS mixed-dtype errors."""
    step = float(step_t.item())
    lr = float(lr_t.item())
    beta1 = float(beta1_t.item())
    beta2 = float(beta2_t.item())
    eps = float(eps_t.item())
    wd = float(wd_t.item())

    # Work in float32 for numerical stability
    p_f32 = p.float()
    grad_f32 = grad.float()

    p_f32.mul_(1 - lr * wd)
    exp_avg.mul_(beta1).add_(grad_f32, alpha=(1 - beta1))
    exp_avg_sq.mul_(beta2).add_(grad_f32.square(), alpha=(1 - beta2))
    bias1 = 1 - beta1 ** step
    bias2 = 1 - beta2 ** step
    denom = (exp_avg_sq / bias2).sqrt() + eps
    step_size = lr / bias1
    p_f32.add_(exp_avg / denom, alpha=-step_size)
    p.copy_(p_f32.to(p.dtype))


def muon_step(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
              momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim):
    """Muon update step with Newton-Schulz orthogonalization, MPS-compatible."""
    # Move scalar tensors to device
    momentum_t = momentum_t.to(device=stacked_grads.device, dtype=stacked_grads.dtype)
    lr_t = lr_t.to(device=stacked_grads.device, dtype=stacked_grads.dtype)
    wd_t = wd_t.to(device=stacked_grads.device, dtype=stacked_grads.dtype)
    beta2_t = beta2_t.to(device=stacked_grads.device, dtype=stacked_grads.dtype)

    # Nesterov momentum — ensure matching dtypes for MPS
    momentum_f = float(momentum_t.item())
    momentum_buffer.mul_(momentum_f).add_(stacked_grads, alpha=(1 - momentum_f))
    g = stacked_grads * (1 - momentum_f) + momentum_buffer * momentum_f

    # Polar express orthogonalization
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X

    # NorMuon variance reduction — all in float32 to avoid MPS mixed-dtype errors
    g_f32 = g.float()
    beta2_f32 = beta2_t.float().to(device=g.device)
    v_mean = g_f32.square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    # second_momentum_buffer is in param dtype; cast to f32 for computation
    smb_f32 = second_momentum_buffer.float()
    smb_f32 = smb_f32 + (1 - beta2_f32) * (v_mean - smb_f32)
    second_momentum_buffer.copy_(smb_f32.to(second_momentum_buffer.dtype))
    step_size = smb_f32.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = (g_f32 * final_scale).to(g.dtype)

    # Cautious weight decay + parameter update — cast to same dtype
    g_param_dtype = g.to(stacked_params.dtype)
    lr_val = lr_t.to(device=stacked_params.device, dtype=stacked_params.dtype)
    wd_val = wd_t.to(device=stacked_params.device, dtype=stacked_params.dtype)
    mask = (g_param_dtype * stacked_params) >= 0
    stacked_params.sub_(lr_val * g_param_dtype + lr_val * wd_val * stacked_params * mask)


# ---------------------------------------------------------------------------
# MuonAdamW Optimizer
# ---------------------------------------------------------------------------

class MuonAdamW(torch.optim.Optimizer):
    """Combined optimizer: Muon for 2D matrix params, AdamW for others.
    Adapted for MPS backend (no torch.compile, device-aware scalar tensors)."""

    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        # 0-D tensors for scalar parameters (kept on CPU, moved to device in step functions)
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _step_adamw(self, group):
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            state['step'] += 1
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            adamw_step(p, grad, state['exp_avg'], state['exp_avg_sq'],
                       self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                       self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t)

    def _step_muon(self, group):
        params = group['params']
        if not params:
            return
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_wd_t.fill_(group["weight_decay"])
        muon_step(stacked_grads, stacked_params,
                  state["momentum_buffer"], state["second_momentum_buffer"],
                  self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t,
                  self._muon_beta2_t, group["ns_steps"], red_dim)
        # Copy updated params back
        for i, param in enumerate(params):
            param.data.copy_(stacked_params[i])

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                self._step_adamw(group)
            elif group['kind'] == 'muon':
                self._step_muon(group)
