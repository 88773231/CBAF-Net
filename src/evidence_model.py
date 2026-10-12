"""Experimental, graph-free temporal evidence model, not the v1.1.0 method."""

import torch
from torch import nn


class TemporalEvidenceModel(nn.Module):
    """Align step-specific hashes, model text/action interaction, and gate experts."""

    def __init__(
        self, description_dim, text_dim, action_dim, tabular_dim, class_count,
        steps=5, hidden_dim=64, dropout=0.2, interaction=True, gate=True,
    ):
        super().__init__()
        if min(description_dim, text_dim, action_dim, tabular_dim,
               class_count, steps, hidden_dim) < 1:
            raise ValueError("model dimensions must be positive")
        self.steps = steps
        self.class_count = class_count
        self.interaction = interaction
        self.use_gate = gate

        def encoder(input_dim):
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim),
                nn.GELU(), nn.Dropout(dropout),
            )

        self.description_encoder = encoder(description_dim)
        self.statistics_encoder = encoder(tabular_dim)
        # The frozen exporter uses a different hashing namespace at each step.
        # Separate projections align those coordinate systems before the GRU.
        self.text_encoders = nn.ModuleList([encoder(text_dim) for _ in range(steps)])
        self.action_encoders = nn.ModuleList([encoder(action_dim) for _ in range(steps)])
        self.cross_view = encoder(hidden_dim * (4 if interaction else 2))
        self.temporal_encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.expert_heads = nn.ModuleList(
            [nn.Linear(hidden_dim, class_count) for _ in range(2)]
        )
        self.gate = (
            nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(hidden_dim, class_count * 2),
            )
            if gate else None
        )

    def forward(self, description, text, action, tabular):
        if text.ndim != 3 or action.ndim != 3:
            raise ValueError("text/action inputs require [rows, steps, features]")
        if text.shape[1] != self.steps or action.shape[1] != self.steps:
            raise ValueError("temporal step count does not match the model")
        if not (description.shape[0] == text.shape[0] == action.shape[0] == tabular.shape[0]):
            raise ValueError("modality rows do not align")
        description_context = self.description_encoder(description)
        pairs = []
        for step in range(self.steps):
            left = self.text_encoders[step](text[:, step])
            right = self.action_encoders[step](action[:, step])
            parts = [left, right]
            if self.interaction:
                parts.extend([left * right, torch.abs(left - right)])
            pairs.append(self.cross_view(torch.cat(parts, dim=1)) + description_context)
        _, hidden = self.temporal_encoder(torch.stack(pairs, dim=1))
        experts = [hidden[-1], self.statistics_encoder(tabular)]
        branch_logits = torch.stack(
            [head(value) for head, value in zip(self.expert_heads, experts)], dim=2
        )
        if self.gate is None:
            weights = torch.full_like(branch_logits, 1.0 / 2.0)
        else:
            weights = self.gate(torch.cat([description_context, *experts], dim=1)).reshape(
                -1, self.class_count, 2
            ).softmax(dim=2)
        return {
            "logits": (weights * branch_logits).sum(dim=2),
            "branch_logits": branch_logits,
            "gate_weights": weights,
        }


class MatchedFeatureMLP(nn.Module):
    """Ordinary supervised baseline on the same description/text/action/statistics."""

    def __init__(self, input_dim, class_count, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, class_count),
        )

    def forward(self, description, text, action, tabular):
        inputs = torch.cat(
            [description, text.flatten(1), action.flatten(1), tabular], dim=1
        )
        return {"logits": self.network(inputs)}
