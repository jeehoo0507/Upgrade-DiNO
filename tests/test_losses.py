"""Fixed vs standard soft-target CE: values, gradients vs float64, zero sum, extreme DINO-like logits."""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dino.losses import DINOLoss, fixed_log_softmax, grad_rel_error, reference_grad, soft_ce  # noqa: E402


def test_log_softmax_and_value():
    torch.manual_seed(0)
    z = torch.randn(8, 1000, dtype=torch.float64) * 5
    assert torch.allclose(fixed_log_softmax(z), F.log_softmax(z, -1), atol=1e-12)
    zs, zt = torch.randn(8, 1000, dtype=torch.float64) * 3, torch.randn(8, 1000, dtype=torch.float64) * 3
    assert torch.allclose(soft_ce(zs, zt, "fixed"), soft_ce(zs, zt, "standard"), atol=1e-10)


def test_fixed_grad_matches_float64_autograd():
    torch.manual_seed(1)
    zs = (torch.randn(6, 500, dtype=torch.float64) * 4).requires_grad_(True)
    zt = torch.randn(6, 500, dtype=torch.float64) * 4
    g_std, = torch.autograd.grad(soft_ce(zs, zt, "standard").sum(), zs)
    g_fix, = torch.autograd.grad(soft_ce(zs, zt, "fixed").sum(), zs)
    assert torch.allclose(g_fix, g_std, atol=1e-12), (g_fix - g_std).abs().max()
    assert torch.allclose(reference_grad(zs, zt), g_std, atol=1e-12)


def test_exact_agreement_gives_zero_gradient():
    torch.manual_seed(2)
    z = torch.randn(4, 4096) * 10
    zs = z.clone().requires_grad_(True)
    g_fix, = torch.autograd.grad(soft_ce(zs, z, "fixed").sum(), zs)
    assert torch.count_nonzero(g_fix) == 0
    zs2 = z.clone().requires_grad_(True)
    g_std, = torch.autograd.grad(soft_ce(zs2, z, "standard").sum(), zs2)
    print(f"  z_S == z_T: standard |g|max {g_std.abs().max():.2e} (should be 0), fixed 0")


def test_near_agreement_error_fp32():
    torch.manual_seed(3)
    zt = torch.randn(16, 65536) * 3 / 0.04 * 0.1          # DINO-like spread
    zs = zt + 1e-3 * torch.randn_like(zt)
    e_std, s_std = grad_rel_error(zs, zt, "standard")
    e_fix, s_fix = grad_rel_error(zs, zt, "fixed")
    print(f"  near agreement fp32 rel err: standard {e_std.median():.2e}, fixed {e_fix.median():.2e}; "
          f"|sum g| standard {s_std.median():.1e}, fixed {s_fix.median():.1e}")
    assert e_fix.median() < e_std.median()


def test_extreme_dino_logits_finite_and_accurate():
    torch.manual_seed(4)
    for tau_t in (0.04, 0.10):
        t = torch.randn(32, 65536) * 2
        zt = t / tau_t                                       # gaps of hundreds
        zs = torch.randn(32, 65536) * 2 / 0.1               # unrelated student, far from teacher
        for impl in ("standard", "fixed"):
            for dt in (torch.float32, torch.bfloat16):
                e, s = grad_rel_error(zs, zt, impl, dt)
                assert torch.isfinite(e).all(), (impl, dt, tau_t)
        e_fix, _ = grad_rel_error(zs, zt, "fixed")
        e_std, _ = grad_rel_error(zs, zt, "standard")
        print(f"  tau_t {tau_t}: far fp32 rel err standard {e_std.median():.2e}, fixed {e_fix.median():.2e}")
        assert e_fix.median() < 1e-4
        # one-hot-ish teacher, student close: large margin regime
        zs2 = zt + torch.randn_like(zt) * 0.5
        e2, _ = grad_rel_error(zs2, zt, "fixed")
        assert torch.isfinite(e2).all() and e2.median() < 1e-3, e2.median()


def test_dino_loss_backward_and_center():
    torch.manual_seed(5)
    for impl in ("standard", "fixed"):
        crit = DINOLoss(256, ncrops=4, impl=impl, teacher_temp=0.04, nepochs=3)
        s = torch.randn(4 * 8, 256, requires_grad=True)
        t = torch.randn(2 * 8, 256)
        loss = crit(s, t, 0)
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(s.grad).all()
        assert crit.center.abs().sum() > 0 and "teacher_entropy" in crit.last
    c1 = DINOLoss(64, 4, "standard", 0.07, warmup_temp=0.04, warmup_epochs=2, nepochs=4)
    assert all(abs(a - b) < 1e-12 for a, b in zip(c1.teacher_temp_schedule, [0.04, 0.055, 0.07, 0.07]))
    s = torch.randn(32, 256, dtype=torch.float64, requires_grad=True)
    t = torch.randn(16, 256, dtype=torch.float64)
    a = DINOLoss(256, 4, "standard", 0.04).double()
    b = DINOLoss(256, 4, "fixed", 0.04).double()
    la, lb = a(s, t, 0), b(s, t, 0)
    ga, = torch.autograd.grad(la, s)
    gb, = torch.autograd.grad(lb, s)
    assert torch.allclose(la, lb, atol=1e-10) and torch.allclose(ga, gb, atol=1e-12)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok   {name}")
    print("loss tests passed")
