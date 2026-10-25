"""DINO loss with two cross-entropy implementations, and the gradient-error probe.

DINO loss for one (teacher view, student view) pair: soft-target cross-entropy
    L = - sum_k p_k log q_k,   p = softmax(z_T),  q = softmax(z_S),
    z_T = (teacher_out - center) / tau_t,   z_S = student_out / tau_s.
dL/dz_S = q - p (p sums to 1).

standard  exactly DINO's code: p = F.softmax(z_T), loss = sum(-p * F.log_softmax(z_S)), gradient by autograd
          (= q * sum(p) - p). Problems (Barron 2026): "softmax cancellation" -- q - p is a difference of nearly
          equal numbers near teacher-student agreement, and forward/backward use different op sequences, so
          the gradient is not zero even when z_S == z_T; at large logit gaps exp() terms round away.
fixed     Barron, "A Fix for Softmax and Cross-Entropy" (arXiv 2610.31642), soft-target form:
          log_softmax with the max entry pinned to exactly 0 and the rest folded into softplus(-m), and a custom
          gradient computed from delta = z_S - z_T without subtracting probabilities:
              d = delta - sum_i p_i delta_i
              s = d - log1p(sum_i p_i expm1(d_i))         (= log q/p)
              u = p * expm1(s)                             (= q - p)
              g = u - q * sum_i u_i                        (exact zero sum)
          Two guards (not in the paper, needed for DINO's 65k-way head at tau_t = 0.04 where logit gaps reach
          hundreds): rows with max|d| >= D_SWITCH use the equivalent s = d - logsumexp(log p + d) (expm1 would
          overflow), and entries with s >= S_SWITCH use u = q - p directly (p underflows while q does not; no
          cancellation there). Both branches are exact in real arithmetic; tests check them against float64.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

D_SWITCH = 20.0
S_SWITCH = 20.0
LOSSES = ("standard", "fixed")


# ------------------------------------------------------------------------------------- primitives ---
def fixed_log_softmax(z, dim=-1):
    """log_softmax with the max entry's shifted logit exactly 0: z~ = z - z_max (0 at argmax),
    m = -LSE_{j != argmax}(z~_j), log_softmax = z~ - softplus(-m)."""
    i = z.argmax(dim, keepdim=True)
    zt = (z - z.gather(dim, i)).scatter(dim, i, 0.0)
    m = -torch.logsumexp(zt.scatter(dim, i, float("-inf")), dim, keepdim=True)
    return zt - F.softplus(-m, beta=1.0, threshold=1e4)   # threshold: no linear shortcut, exact log1p(exp)


def _soft_grad(zs, zt, logp, p, q):
    """dL/dz_S = q - p computed without subtracting probabilities (see module docstring)."""
    delta = zs - zt
    d = delta - (p * delta).sum(-1, keepdim=True)
    near = d.abs().amax(-1, keepdim=True) < D_SWITCH
    s_near = d - torch.log1p((p * torch.expm1(d.clamp(max=D_SWITCH))).sum(-1, keepdim=True))
    s_far = d - torch.logsumexp(logp + d, -1, keepdim=True)
    s = torch.where(near, s_near, s_far)
    u = torch.where(s < S_SWITCH, p * torch.expm1(s.clamp(max=S_SWITCH)), q - p)
    return u - q * u.sum(-1, keepdim=True)


class _FixedSoftCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, zs, zt):
        logp = fixed_log_softmax(zt)
        p = logp.exp()
        logq = fixed_log_softmax(zs)
        q = logq.exp()
        loss = -(p * logq).sum(-1)
        ctx.save_for_backward(_soft_grad(zs, zt, logp, p, q))
        return loss

    @staticmethod
    def backward(ctx, gout):
        (g,) = ctx.saved_tensors
        return gout.unsqueeze(-1) * g, None


def soft_ce(zs, zt, impl):
    """Per-sample soft-target CE between student logits zs and teacher logits zt (teacher gets no gradient)."""
    zt = zt.detach()
    if impl == "standard":
        return torch.sum(-F.softmax(zt, dim=-1) * F.log_softmax(zs, dim=-1), dim=-1)   # DINO main_dino.py
    if impl == "fixed":
        return _FixedSoftCE.apply(zs, zt)
    raise ValueError(impl)


@torch.no_grad()
def reference_grad(zs, zt):
    """float64 reference for dL/dz_S, using the fixed formulas in float64 (more accurate than q - p in float64)."""
    zs, zt = zs.double(), zt.double()
    logp = fixed_log_softmax(zt)
    return _soft_grad(zs, zt, logp, logp.exp(), fixed_log_softmax(zs).exp())


def grad_rel_error(zs, zt, impl, dtype=torch.float32):
    """Per-sample relative L2 error of dL/dz_S computed by `impl` in `dtype`, vs the float64 reference.
    Returns (rel_err (B,), |sum_k g_k| (B,)) as float64 CPU tensors."""
    ref = reference_grad(zs, zt)
    x = zs.detach().to(dtype).requires_grad_(True)
    with torch.enable_grad():
        loss = soft_ce(x, zt.detach().to(dtype), impl).sum()
        (g,) = torch.autograd.grad(loss, x)
    g = g.double()
    err = (g - ref).norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-300)
    return err.cpu(), g.sum(-1).abs().cpu()


# -------------------------------------------------------------------------------------- DINO loss ---
class DINOLoss(nn.Module):
    """facebookresearch/dino main_dino.py DINOLoss with a selectable cross-entropy implementation.
    Teacher temperature: constant unless warmup_epochs > 0 (then linear warmup_temp -> teacher_temp)."""

    def __init__(self, out_dim, ncrops, impl, teacher_temp, student_temp=0.1, center_momentum=0.9,
                 warmup_temp=0.04, warmup_epochs=0, nepochs=100):
        super().__init__()
        assert impl in LOSSES, impl
        self.impl, self.student_temp, self.center_momentum, self.ncrops = impl, student_temp, center_momentum, ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        sched = [warmup_temp + (teacher_temp - warmup_temp) * e / warmup_epochs for e in range(warmup_epochs)]
        self.teacher_temp_schedule = (sched + [teacher_temp] * nepochs)[:nepochs]
        self.last = {}

    def logits(self, student_output, teacher_output, epoch):
        temp = self.teacher_temp_schedule[epoch]
        zs = (student_output / self.student_temp).chunk(self.ncrops)
        zt = ((teacher_output - self.center) / temp).detach().chunk(2)
        return zs, zt

    def forward(self, student_output, teacher_output, epoch):
        zs, zt = self.logits(student_output, teacher_output, epoch)
        total, n = 0.0, 0
        for iq, t in enumerate(zt):
            for v, s in enumerate(zs):
                if v == iq:
                    continue          # skip the same view
                total = total + soft_ce(s, t, self.impl).mean()
                n += 1
        with torch.no_grad():
            p = F.softmax(zt[0].float(), dim=-1)
            self.last = {"teacher_entropy": float(-(p * p.clamp_min(1e-30).log()).sum(-1).mean()),
                         "teacher_maxprob": float(p.max(-1).values.mean())}
        self.update_center(teacher_output)
        return total / n

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = teacher_output.mean(0, keepdim=True)        # single GPU (no all_reduce)
        self.center.mul_(self.center_momentum).add_(batch_center, alpha=1 - self.center_momentum)
