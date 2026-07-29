"""Joint training objective used for the published OCQT configurations.

The objective combines weighted eight-class cross-entropy, weighted
season-presence binary cross-entropy, rice-season-count cross-entropy, and a
consistency loss between class-derived and query-derived season probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


CLASS_TO_SEASON_BITS = torch.tensor(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ],
    dtype=torch.float32,
)

CLASS_TO_COUNT = torch.tensor(
    [0, 1, 1, 1, 2, 2, 2, 3],
    dtype=torch.long,
)


@dataclass(frozen=True)
class OCQTLossConfig:
    """Loss coefficients used in the manuscript experiments."""

    season_loss_weight: float = 0.12
    count_loss_weight: float = 0.08
    consistency_loss_weight: float = 0.03
    label_smoothing: float = 0.05


def weighted_batch_mean(
    values: torch.Tensor,
    sample_weight: torch.Tensor,
) -> torch.Tensor:
    """Return a numerically stable sample-weighted batch mean."""
    weight = sample_weight.to(dtype=values.dtype).reshape(-1)
    if values.ndim != 1 or values.shape[0] != weight.shape[0]:
        raise ValueError("values and sample_weight must be one-dimensional and aligned")
    return (values * weight).sum() / weight.sum().clamp_min(1e-8)


def compute_ocqt_loss(
    output: Dict[str, torch.Tensor],
    class_target: torch.Tensor,
    season_target: torch.Tensor,
    sample_weight: torch.Tensor,
    *,
    class_weight: Optional[torch.Tensor] = None,
    season_pos_weight: Optional[torch.Tensor] = None,
    config: OCQTLossConfig | None = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute the joint OCQT loss.

    Parameters
    ----------
    output:
        Dictionary returned by ``OrderedCycleQueryTransformer``.
    class_target:
        Integer class labels with shape ``[batch]``.
    season_target:
        Early/middle/late presence targets with shape ``[batch, 3]``.
    sample_weight:
        Per-sample weights with shape ``[batch]``. The manuscript used 1.0 for
        real samples and 0.25 for high-quality PTS samples in OCQT (PTS0.25).
    class_weight:
        Optional eight-class weight tensor.
    season_pos_weight:
        Optional three-element positive-class weight tensor.
    config:
        Loss coefficients and label smoothing.
    """
    cfg = config or OCQTLossConfig()
    device = class_target.device

    class_loss_each = F.cross_entropy(
        output["class_logits"],
        class_target,
        weight=class_weight,
        label_smoothing=cfg.label_smoothing,
        reduction="none",
    )

    season_loss_each = F.binary_cross_entropy_with_logits(
        output["season_logits"],
        season_target,
        pos_weight=season_pos_weight,
        reduction="none",
    ).mean(dim=1)

    count_lookup = CLASS_TO_COUNT.to(device=device)
    count_target = count_lookup[class_target]
    count_loss_each = F.cross_entropy(
        output["count_logits"],
        count_target,
        reduction="none",
    )

    class_probability = torch.softmax(output["class_logits"], dim=1)
    season_bits = CLASS_TO_SEASON_BITS.to(
        device=class_probability.device,
        dtype=class_probability.dtype,
    )
    season_from_class = class_probability @ season_bits
    season_from_query = torch.sigmoid(output["season_logits"])
    consistency_loss_each = (
        season_from_query - season_from_class
    ).pow(2).mean(dim=1)

    total_each = (
        class_loss_each
        + cfg.season_loss_weight * season_loss_each
        + cfg.count_loss_weight * count_loss_each
        + cfg.consistency_loss_weight * consistency_loss_each
    )

    losses = {
        "total_loss": weighted_batch_mean(total_each, sample_weight),
        "class_loss": weighted_batch_mean(class_loss_each, sample_weight),
        "season_loss": weighted_batch_mean(season_loss_each, sample_weight),
        "count_loss": weighted_batch_mean(count_loss_each, sample_weight),
        "consistency_loss": weighted_batch_mean(
            consistency_loss_each,
            sample_weight,
        ),
    }
    return losses["total_loss"], losses


__all__ = [
    "CLASS_TO_SEASON_BITS",
    "CLASS_TO_COUNT",
    "OCQTLossConfig",
    "weighted_batch_mean",
    "compute_ocqt_loss",
]
