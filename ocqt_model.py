"""Core implementation of the Ordered-Cycle Query Transformer (OCQT).

OCQT encodes one annual-global feature representation, up to three
chronologically ordered cycle representations, one learnable classification
token, and three learnable season-query tokens. The model produces logits for
three related tasks:

1. eight-class rice cropping-system classification;
2. early-/middle-/late-season presence prediction; and
3. rice-season-count prediction (0, 1, 2, or 3 seasons).

The public implementation expects precomputed annual-global and ordered-cycle
features. Feature construction, training data, fitted scalers, and trained
weights are not included in this repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import nn


@dataclass(frozen=True)
class OCQTConfig:
    """Architecture settings used in the manuscript experiments."""

    d_model: int = 64
    n_heads: int = 4
    ff_dim: int = 128
    depth: int = 3
    dropout: float = 0.20
    attention_dropout: float = 0.10
    low_quality_token_dropout: float = 0.10
    num_classes: int = 8
    num_season_queries: int = 3
    num_count_classes: int = 4


class PreNormTransformerBlock(nn.Module):
    """Pre-normalized Transformer encoder block with masked token support."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_dim: int,
        dropout: float,
        attention_dropout: float,
    ) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        z = self.norm1(x)
        attn_out, attn_weights = self.attn(
            z,
            z,
            z,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            average_attn_weights=False,
        )
        x = x + self.drop1(attn_out)
        x = x + self.ff(self.norm2(x))

        # Keep padded ordered-cycle positions equal to zero after each block.
        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        return x, attn_weights if need_weights else None


class OrderedCycleQueryTransformer(nn.Module):
    """Ordered-Cycle Query Transformer for eight-class classification.

    Parameters
    ----------
    global_dim:
        Dimension of the annual-global feature representation.
    cycle_dim:
        Dimension of each ordered-cycle feature representation.
    config:
        OCQT architecture configuration.

    Input shapes
    ------------
    annual_global_features: ``[batch, global_dim]``
    ordered_cycle_features: ``[batch, 3, cycle_dim]``
    cycle_quality: ``[batch, 3]``
    cycle_mask: ``[batch, 3]`` where ``True`` marks a missing cycle position.

    Output dictionary
    -----------------
    class_logits: ``[batch, 8]``
    season_logits: ``[batch, 3]``
    count_logits: ``[batch, 4]``
    query_cycle_attention: optional ``[batch, 3, 3]``
    """

    CLS_POS = 0
    GLOBAL_POS = 1
    CYCLE_SLICE = slice(2, 5)
    QUERY_SLICE = slice(5, 8)

    def __init__(
        self,
        global_dim: int,
        cycle_dim: int,
        config: OCQTConfig | None = None,
    ) -> None:
        super().__init__()

        if global_dim <= 0 or cycle_dim <= 0:
            raise ValueError("global_dim and cycle_dim must be positive integers")

        self.config = config or OCQTConfig()
        cfg = self.config
        d = cfg.d_model

        if cfg.num_season_queries != 3:
            raise ValueError("The published OCQT architecture uses three season queries")

        self.global_projector = nn.Sequential(
            nn.Linear(global_dim, cfg.ff_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.ff_dim, d),
            nn.LayerNorm(d),
        )

        self.cycle_projector = nn.Sequential(
            nn.Linear(cycle_dim, cfg.ff_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.ff_dim, d),
            nn.LayerNorm(d),
        )

        self.classification_token = nn.Parameter(torch.zeros(1, 1, d))
        self.season_queries = nn.Parameter(torch.zeros(1, 3, d))

        # Token types: classification / annual-global / ordered-cycle / query.
        self.type_embedding = nn.Embedding(4, d)
        self.cycle_order_embedding = nn.Parameter(torch.zeros(1, 3, d))

        self.blocks = nn.ModuleList(
            [
                PreNormTransformerBlock(
                    d_model=d,
                    n_heads=cfg.n_heads,
                    ff_dim=cfg.ff_dim,
                    dropout=cfg.dropout,
                    attention_dropout=cfg.attention_dropout,
                )
                for _ in range(cfg.depth)
            ]
        )
        self.final_norm = nn.LayerNorm(d)

        # The primary head uses the contextualized representations of all eight tokens.
        self.classifier = nn.Sequential(
            nn.LayerNorm(d * 8),
            nn.Linear(d * 8, 128),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(64, cfg.num_classes),
        )

        # One binary logit is produced from each season-query token.
        self.season_head = nn.Linear(d, 1)

        # The count head uses classification + annual-global + three cycle tokens.
        self.count_head = nn.Sequential(
            nn.LayerNorm(d * 5),
            nn.Linear(d * 5, 64),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(64, cfg.num_count_classes),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.classification_token, std=0.02)
        nn.init.trunc_normal_(self.season_queries, std=0.02)
        nn.init.trunc_normal_(self.cycle_order_embedding, std=0.02)

    @staticmethod
    def _validate_inputs(
        annual_global_features: torch.Tensor,
        ordered_cycle_features: torch.Tensor,
        cycle_quality: torch.Tensor,
        cycle_mask: torch.Tensor,
    ) -> None:
        if annual_global_features.ndim != 2:
            raise ValueError("annual_global_features must have shape [batch, global_dim]")
        if ordered_cycle_features.ndim != 3 or ordered_cycle_features.shape[1] != 3:
            raise ValueError(
                "ordered_cycle_features must have shape [batch, 3, cycle_dim]"
            )
        if cycle_quality.shape != ordered_cycle_features.shape[:2]:
            raise ValueError("cycle_quality must have shape [batch, 3]")
        if cycle_mask.shape != ordered_cycle_features.shape[:2]:
            raise ValueError("cycle_mask must have shape [batch, 3]")
        if annual_global_features.shape[0] != ordered_cycle_features.shape[0]:
            raise ValueError("All inputs must have the same batch dimension")
        if cycle_mask.dtype != torch.bool:
            raise TypeError("cycle_mask must be a boolean tensor")

    def _apply_lower_quality_token_dropout(
        self,
        cycle_quality: torch.Tensor,
        cycle_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the training-time masking procedure used in the manuscript."""
        effective_mask = cycle_mask.clone()
        probability = self.config.low_quality_token_dropout

        if not self.training or probability <= 0:
            return effective_mask

        valid = ~cycle_mask
        if not torch.any(valid):
            return effective_mask

        quality_median = torch.median(cycle_quality[valid])
        lower_quality = valid & (cycle_quality <= quality_median)
        random_drop = torch.rand_like(cycle_quality) < probability
        return effective_mask | (lower_quality & random_drop)

    def forward(
        self,
        annual_global_features: torch.Tensor,
        ordered_cycle_features: torch.Tensor,
        cycle_quality: torch.Tensor,
        cycle_mask: torch.Tensor,
        need_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        self._validate_inputs(
            annual_global_features,
            ordered_cycle_features,
            cycle_quality,
            cycle_mask,
        )

        batch_size = annual_global_features.shape[0]
        effective_cycle_mask = self._apply_lower_quality_token_dropout(
            cycle_quality,
            cycle_mask,
        )

        classification_token = self.classification_token.expand(batch_size, -1, -1)
        annual_global_token = self.global_projector(
            annual_global_features
        ).unsqueeze(1)
        ordered_cycle_tokens = (
            self.cycle_projector(ordered_cycle_features)
            + self.cycle_order_embedding
        )
        season_query_tokens = self.season_queries.expand(batch_size, -1, -1)

        classification_token = (
            classification_token + self.type_embedding.weight[0][None, None, :]
        )
        annual_global_token = (
            annual_global_token + self.type_embedding.weight[1][None, None, :]
        )
        ordered_cycle_tokens = (
            ordered_cycle_tokens + self.type_embedding.weight[2][None, None, :]
        )
        season_query_tokens = (
            season_query_tokens + self.type_embedding.weight[3][None, None, :]
        )

        ordered_cycle_tokens = ordered_cycle_tokens.masked_fill(
            effective_cycle_mask.unsqueeze(-1),
            0.0,
        )

        token_sequence = torch.cat(
            [
                classification_token,
                annual_global_token,
                ordered_cycle_tokens,
                season_query_tokens,
            ],
            dim=1,
        )

        full_mask = torch.zeros(
            batch_size,
            8,
            dtype=torch.bool,
            device=token_sequence.device,
        )
        full_mask[:, self.CYCLE_SLICE] = effective_cycle_mask

        last_attention: Optional[torch.Tensor] = None
        encoded = token_sequence
        for block_index, block in enumerate(self.blocks):
            encoded, attention = block(
                encoded,
                key_padding_mask=full_mask,
                need_weights=(need_attention and block_index == len(self.blocks) - 1),
            )
            if attention is not None:
                last_attention = attention

        encoded = self.final_norm(encoded)

        class_logits = self.classifier(encoded.reshape(batch_size, -1))
        season_logits = self.season_head(encoded[:, self.QUERY_SLICE]).squeeze(-1)
        count_logits = self.count_head(encoded[:, :5].reshape(batch_size, -1))

        output: Dict[str, torch.Tensor] = {
            "class_logits": class_logits,
            "season_logits": season_logits,
            "count_logits": count_logits,
        }

        if last_attention is not None:
            output["query_cycle_attention"] = last_attention.mean(dim=1)[
                :, self.QUERY_SLICE, self.CYCLE_SLICE
            ]

        return output


def build_ocqt(
    global_dim: int,
    cycle_dim: int,
    **config_overrides: object,
) -> OrderedCycleQueryTransformer:
    """Construct OCQT with optional overrides of published hyperparameters."""
    config = OCQTConfig(**config_overrides)
    return OrderedCycleQueryTransformer(
        global_dim=global_dim,
        cycle_dim=cycle_dim,
        config=config,
    )


__all__ = [
    "OCQTConfig",
    "PreNormTransformerBlock",
    "OrderedCycleQueryTransformer",
    "build_ocqt",
]
