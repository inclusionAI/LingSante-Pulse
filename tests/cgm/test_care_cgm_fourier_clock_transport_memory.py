import numpy as np, torch
from lingsante_pulse.cgm.models.memory import (
    FourierClockTransportMemory,
    controlled_base_rows,
    parameter_budget,
    spectral_rows,
)


def fixture():
    torch.manual_seed(43)
    base = torch.randn(2, 4, 12, 8)
    theta = torch.linspace(0, 6, 12)
    clock = (
        torch.stack([theta.sin(), theta.cos()], -1)[None, None]
        .expand(2, 4, 12, 2)
        .clone()
    )
    mask = torch.ones(2, 4, 12, dtype=torch.bool)
    return base, clock, mask


def canonical(x):
    x = np.ascontiguousarray(x)
    return np.sort(
        x.view(np.dtype((np.void, x.dtype.itemsize * x.shape[-1]))).reshape(-1)
    )


def test_control_preserves_canonical_rows_bitwise():
    base, _, mask = fixture()
    control = controlled_base_rows(base, mask, True)
    assert np.array_equal(canonical(base.numpy()), canonical(control.numpy()))
    assert not torch.equal(base, control)


def test_physical_fourier_interaction_changes():
    base, clock, mask = fixture()
    assert not torch.equal(
        spectral_rows(base, clock, mask, False), spectral_rows(base, clock, mask, True)
    )


def test_equal_budget_zero_init_and_146d():
    a = FourierClockTransportMemory(False)
    b = FourierClockTransportMemory(True)
    assert parameter_budget()["final_dimension"] == 146 and sum(
        p.numel() for p in a.parameters()
    ) == sum(p.numel() for p in b.parameters())
    assert int((a.fctm_writeback.weight != 0).sum()) == 0
