import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class StructuralAttention(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.2):
        super(StructuralAttention, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.qk = nn.Linear(in_dim, out_dim)
        self.value = nn.Linear(in_dim, out_dim)
        self.output_proj = nn.Linear(out_dim, out_dim)
        self.layernorm = nn.LayerNorm(out_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.1)

    def forward(self, x, edge_index):
        residual = x
        qk = self.qk(x)
        q, k = qk, qk
        v = self.value(x)

        if edge_index.size(1) > 0:
            src, dst = edge_index
            valid_edges = (
                (src >= 0)
                & (dst >= 0)
                & (src < x.size(0))
                & (dst < x.size(0))
            )
            src, dst = src[valid_edges], dst[valid_edges]

            if len(src) > 0:
                q_dst, k_src = q[dst], k[src]
                scores = torch.sum(q_dst * k_src, dim=-1) / math.sqrt(self.out_dim)
                scores = torch.clamp(scores, -5.0, 5.0)
                max_per_dst = torch.full(
                    (x.size(0),),
                    -torch.inf,
                    device=scores.device,
                    dtype=scores.dtype,
                )
                max_per_dst.scatter_reduce_(
                    0,
                    dst,
                    scores,
                    reduce="amax",
                    include_self=True,
                )
                exp_scores = torch.exp(scores - max_per_dst[dst])
                denom = torch.zeros(
                    x.size(0),
                    device=scores.device,
                    dtype=scores.dtype,
                )
                denom.scatter_add_(0, dst, exp_scores)
                attn_weights = exp_scores / denom[dst].clamp_min(1e-12)
                attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
                out = torch.zeros_like(x)
                out.index_add_(0, dst, attn_weights.unsqueeze(1) * v[src])
            else:
                out = v
        else:
            out = v

        out = self.output_proj(out)
        out = self.layernorm(out + residual)
        return out


class TemporalMeanPooling(nn.Module):
    """Mean-pool temporal steps, then apply normalization and dropout."""

    def __init__(self, embedding_dim, dropout=0.2):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.layernorm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        pooled = x.mean(dim=1)
        pooled = self.layernorm(pooled)
        pooled = self.dropout(pooled)
        return pooled
