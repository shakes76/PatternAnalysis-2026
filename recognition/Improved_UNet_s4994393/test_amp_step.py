"""Exercise real GradScaler overflow recovery on CPU without requiring CUDA."""

import torch
from train import optimizer_step


def main():
    """Inject overflow, verify skipped weights, then verify a finite recovery step."""
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler("cpu", init_scale=128)
    original = [p.detach().clone() for p in model.parameters()]
    scaler.scale(model(torch.ones(1, 2)).sum()).backward()
    # Simulate a numerical overflow after backward, before the optimizer can update.
    model.weight.grad.fill_(float("inf"))
    assert not optimizer_step(model, optimizer, scaler)
    assert scaler.get_scale() == 64
    assert not optimizer.state
    assert all(torch.equal(p, before) for p, before in zip(model.parameters(), original))
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(model(torch.ones(1, 2)).sum()).backward()
    assert optimizer_step(model, optimizer, scaler)
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert any(not torch.equal(p, before) for p, before in zip(model.parameters(), original))
    optimizer.zero_grad(set_to_none=True)
    # Without scaling there is no recovery mechanism: invalid gradients must raise.
    disabled = torch.amp.GradScaler("cpu", enabled=False)
    model(torch.ones(1, 2)).sum().backward()
    model.weight.grad.fill_(float("nan"))
    try:
        optimizer_step(model, optimizer, disabled)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Non-AMP non-finite gradients must still raise")
    print("AMP overflow skip, scale reduction, recovery and non-AMP checks passed")


if __name__ == "__main__":
    main()
