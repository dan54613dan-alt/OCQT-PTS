"""Minimal forward-pass and loss smoke test for OCQT."""

from __future__ import annotations

import torch

from ocqt_model import OrderedCycleQueryTransformer
from ocqt_objective import CLASS_TO_SEASON_BITS, compute_ocqt_loss


def main() -> None:
    torch.manual_seed(42)
    batch_size = 4
    global_dim = 38
    cycle_dim = 97

    model = OrderedCycleQueryTransformer(
        global_dim=global_dim,
        cycle_dim=cycle_dim,
    )
    model.eval()

    annual_global_features = torch.randn(batch_size, global_dim)
    ordered_cycle_features = torch.randn(batch_size, 3, cycle_dim)
    cycle_quality = torch.rand(batch_size, 3)
    cycle_mask = torch.tensor(
        [
            [False, False, True],
            [False, True, True],
            [False, False, False],
            [True, True, True],
        ],
        dtype=torch.bool,
    )

    with torch.no_grad():
        outputs = model(
            annual_global_features,
            ordered_cycle_features,
            cycle_quality,
            cycle_mask,
            need_attention=True,
        )

    for name, tensor in outputs.items():
        print(f"{name}: {tuple(tensor.shape)}")

    class_target = torch.tensor([0, 1, 5, 7], dtype=torch.long)
    season_target = CLASS_TO_SEASON_BITS[class_target]
    sample_weight = torch.tensor([1.0, 1.0, 0.25, 0.25])
    total_loss, losses = compute_ocqt_loss(
        outputs,
        class_target,
        season_target,
        sample_weight,
    )
    print(f"total_loss: {float(total_loss):.6f}")
    print("loss terms:", ", ".join(sorted(losses)))


if __name__ == "__main__":
    main()
