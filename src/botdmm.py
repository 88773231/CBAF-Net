import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers.attention import TemporalAttention
from .layers.constraints import OrthogonalConstraint
from .layers.decoupler import ContentDecoupler, StructureDecoupler


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd=1.0):
    return GradientReverse.apply(x, lambd)


def quadfusion_vector_stats(tensor):
    abs_tensor = tensor.abs()
    denom = max(tensor.shape[1], 1) ** 0.5
    return torch.stack(
        [
            tensor.mean(dim=1),
            tensor.std(dim=1, unbiased=False),
            tensor.min(dim=1).values,
            tensor.max(dim=1).values,
            tensor.norm(p=2, dim=1) / denom,
            abs_tensor.mean(dim=1),
            abs_tensor.max(dim=1).values,
            (tensor > 0).float().mean(dim=1),
        ],
        dim=1,
    )


class ResidualMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        return self.norm(self.net(x) + self.skip(x))


class QuadFusionNet(nn.Module):
    """Paper-facing multi-expert gated model for TwiBot20 and QuadBot."""

    def __init__(
        self,
        view_dim,
        graph_dim,
        hidden_dim,
        num_classes,
        dropout,
        ablation="full",
        num_prop_dim=0,
        llm_feature_dim=0,
        expert_dropout=0.0,
        aux_weight=0.2,
    ):
        super().__init__()
        valid_ablations = {
            "full",
            "no_graph",
            "no_temporal",
            "no_style",
            "no_semantic",
            "uniform_gate",
        }
        if ablation not in valid_ablations:
            raise ValueError(f"Unsupported ablation: {ablation}")
        self.ablation = ablation
        self.num_prop_dim = int(num_prop_dim)
        self.llm_feature_dim = int(llm_feature_dim)
        self.extra_dim = self.num_prop_dim + self.llm_feature_dim
        self.expert_dropout = float(expert_dropout)
        self.aux_weight = float(aux_weight)
        self.semantic_expert = ResidualMLP(view_dim * 3, hidden_dim * 2, hidden_dim, dropout)
        self.graph_expert = ResidualMLP(graph_dim, hidden_dim * 2, hidden_dim, dropout)
        self.temporal_gru = nn.GRU(
            input_size=view_dim * 2,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.temporal_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
        )
        self.style_expert = ResidualMLP(24 + self.extra_dim, hidden_dim, hidden_dim, dropout)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.aux_classifier = nn.Linear(hidden_dim * 4, num_classes)
        self.residual_classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, num_classes),
        )

    def forward(self, des, texts, amrs, graph, num_prop=None, llm_features=None):
        text_mean = texts.mean(dim=1)
        amr_mean = amrs.mean(dim=1)
        semantic = self.semantic_expert(torch.cat([des, text_mean, amr_mean], dim=1))
        graph_h = self.graph_expert(graph)

        temporal_in = torch.cat([texts, amrs], dim=2)
        temporal_out, _ = self.temporal_gru(temporal_in)
        temporal_h = self.temporal_proj(temporal_out.mean(dim=1))

        stats = torch.cat(
            [
                quadfusion_vector_stats(des),
                quadfusion_vector_stats(text_mean),
                quadfusion_vector_stats(amr_mean),
            ],
            dim=1,
        )
        extra_features = []
        if self.num_prop_dim > 0:
            if num_prop is None:
                num_prop = torch.zeros(stats.shape[0], self.num_prop_dim, device=stats.device, dtype=stats.dtype)
            extra_features.append(num_prop)
        if self.llm_feature_dim > 0:
            if llm_features is None:
                llm_features = torch.zeros(stats.shape[0], self.llm_feature_dim, device=stats.device, dtype=stats.dtype)
            extra_features.append(llm_features)
        if extra_features:
            stats = torch.cat([stats] + extra_features, dim=1)
        style = self.style_expert(stats)

        if self.ablation == "no_semantic":
            semantic = torch.zeros_like(semantic)
        if self.ablation == "no_temporal":
            temporal_h = torch.zeros_like(temporal_h)
        if self.ablation == "no_graph":
            graph_h = torch.zeros_like(graph_h)
        if self.ablation == "no_style":
            style = torch.zeros_like(style)

        experts = torch.stack([semantic, temporal_h, graph_h, style], dim=1)
        flat = torch.cat([semantic, temporal_h, graph_h, style], dim=1)
        if self.ablation == "uniform_gate":
            gate = torch.full(
                (flat.shape[0], experts.shape[1]),
                1.0 / experts.shape[1],
                device=flat.device,
                dtype=flat.dtype,
            )
        else:
            gate = torch.softmax(self.gate(flat), dim=1)
        if self.training and self.expert_dropout > 0 and self.ablation == "full":
            keep = (torch.rand_like(gate) > self.expert_dropout).float()
            empty = keep.sum(dim=1, keepdim=True) == 0
            keep = torch.where(empty, torch.ones_like(keep), keep)
            gate = gate * keep
            gate = gate / gate.sum(dim=1, keepdim=True).clamp_min(1e-8)
        fused = (experts * gate.unsqueeze(-1)).sum(dim=1)
        logits = self.classifier(fused) + self.aux_weight * self.aux_classifier(flat)
        logits = logits + 0.1 * self.residual_classifier(flat)
        return logits, gate, fused


class CrossModalFusionBlock(nn.Module):
    """Bidirectional text/AMR interaction used by QuadFusion-Net++."""

    def __init__(self, view_dim, hidden_dim, dropout):
        super().__init__()
        heads = 4 if hidden_dim % 4 == 0 else 1
        self.text_proj = nn.Linear(view_dim, hidden_dim)
        self.amr_proj = nn.Linear(view_dim, hidden_dim)
        self.text_attn = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.amr_attn = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.text_norm = nn.LayerNorm(hidden_dim)
        self.amr_norm = nn.LayerNorm(hidden_dim)
        self.text_ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.amr_ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, texts, amrs):
        text_h = self.text_proj(texts)
        amr_h = self.amr_proj(amrs)
        text_ctx, _ = self.text_attn(text_h, amr_h, amr_h, need_weights=False)
        amr_ctx, _ = self.amr_attn(amr_h, text_h, text_h, need_weights=False)
        text_h = self.text_norm(text_h + text_ctx)
        amr_h = self.amr_norm(amr_h + amr_ctx)
        text_h = self.text_norm(text_h + self.text_ff(text_h))
        amr_h = self.amr_norm(amr_h + self.amr_ff(amr_h))
        return text_h, amr_h


class MultiScaleDualStreamTemporalFusion(nn.Module):
    """1-D adaptation of the supplied BFM idea for tweet/AMR sequences.

    The original BFM is a 2-D image module. Here the two streams are temporal
    sequences, so multi-scale depthwise 1-D convolutions are used instead of
    spatial convolutions. Stream weights are computed from pooled evidence and
    normalized across the tweet and AMR streams.
    """

    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.temporal_convs = nn.ModuleList(
            [
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2, groups=hidden_dim),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2, groups=hidden_dim),
            ]
        )
        self.pointwise = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.stream_gate = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def _multi_scale(self, sequence):
        x = sequence.transpose(1, 2)
        branches = [F.gelu(conv(x)) for conv in self.temporal_convs]
        return self.pointwise(sum(branches) / len(branches)).transpose(1, 2)

    def forward(self, text_h, amr_h):
        text_ms = self._multi_scale(text_h)
        amr_ms = self._multi_scale(amr_h)
        pooled = torch.cat(
            [
                text_ms.mean(dim=1),
                text_ms.max(dim=1).values,
                amr_ms.mean(dim=1),
                amr_ms.max(dim=1).values,
            ],
            dim=1,
        )
        stream_weights = torch.softmax(self.stream_gate(pooled), dim=1)
        fused = (
            stream_weights[:, 0].view(-1, 1, 1) * text_ms
            + stream_weights[:, 1].view(-1, 1, 1) * amr_ms
        )
        fused = fused + 0.20 * (text_ms * amr_ms)
        return self.norm(fused), stream_weights


class QuadFusionNetV2(nn.Module):
    """QuadFusion-Net++: cross-modal, confidence-routed multi-expert fusion.

    The model keeps the four expert decomposition but addresses two weaknesses of
    the original implementation: text and AMR are interacted before pooling, and
    routing is conditioned on each expert's predictive confidence rather than on
    a flat concatenation alone. A cosine prototype head provides a second,
    class-aware decision surface and is trained jointly with the classifier.
    """

    def __init__(
        self,
        view_dim,
        graph_dim,
        hidden_dim,
        num_classes,
        dropout,
        num_prop_dim=0,
        llm_feature_dim=0,
        route_temperature=0.7,
        prototype_temperature=0.12,
    ):
        super().__init__()
        self.num_prop_dim = int(num_prop_dim)
        self.llm_feature_dim = int(llm_feature_dim)
        self.extra_dim = self.num_prop_dim + self.llm_feature_dim
        self.num_classes = int(num_classes)
        self.route_temperature = float(route_temperature)
        self.prototype_temperature = float(prototype_temperature)

        self.cross_modal = CrossModalFusionBlock(view_dim, hidden_dim, dropout)
        self.temporal_fusion = MultiScaleDualStreamTemporalFusion(hidden_dim, dropout)
        self.des_proj = nn.Sequential(
            nn.Linear(view_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.semantic_expert = ResidualMLP(hidden_dim * 4, hidden_dim * 2, hidden_dim, dropout)

        self.temporal_gru = nn.GRU(
            input_size=hidden_dim * 4,
            hidden_size=max(hidden_dim // 2, 1),
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.temporal_score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.temporal_norm = nn.LayerNorm(hidden_dim)

        self.graph_expert = ResidualMLP(graph_dim, hidden_dim * 2, hidden_dim, dropout)
        self.style_expert = ResidualMLP(24 + self.extra_dim, hidden_dim * 2, hidden_dim, dropout)
        # Preserve the original feature evidence as a residual path. The former
        # model routed a heavily compressed style statistic vector, which made
        # Human vs. Traditional Bot separation unnecessarily lossy.
        self.raw_input_dim = view_dim * 3 + graph_dim + self.extra_dim
        self.raw_evidence = ResidualMLP(self.raw_input_dim, hidden_dim * 2, hidden_dim, dropout)
        self.raw_classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        self.expert_heads = nn.ModuleList(
            [nn.Linear(hidden_dim, num_classes) for _ in range(4)]
        )
        self.router = nn.Sequential(
            nn.Linear(hidden_dim * 4 + 4, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4),
        )
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.interaction_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.prototypes = nn.Parameter(torch.empty(num_classes, hidden_dim))
        nn.init.xavier_uniform_(self.prototypes)

    def _style_input(self, des, text_mean, amr_mean, num_prop, llm_features):
        stats = torch.cat(
            [
                quadfusion_vector_stats(des),
                quadfusion_vector_stats(text_mean),
                quadfusion_vector_stats(amr_mean),
            ],
            dim=1,
        )
        extras = []
        if self.num_prop_dim > 0:
            if num_prop is None:
                num_prop = torch.zeros(stats.shape[0], self.num_prop_dim, device=stats.device, dtype=stats.dtype)
            extras.append(num_prop)
        if self.llm_feature_dim > 0:
            if llm_features is None:
                llm_features = torch.zeros(stats.shape[0], self.llm_feature_dim, device=stats.device, dtype=stats.dtype)
            extras.append(llm_features)
        return torch.cat([stats] + extras, dim=1) if extras else stats

    def forward(self, des, texts, amrs, graph, num_prop=None, llm_features=None):
        text_h, amr_h = self.cross_modal(texts, amrs)
        temporal_fused, stream_weights = self.temporal_fusion(text_h, amr_h)
        text_h = text_h + 0.35 * temporal_fused
        amr_h = amr_h + 0.35 * temporal_fused
        text_mean = text_h.mean(dim=1)
        amr_mean = amr_h.mean(dim=1)
        des_h = self.des_proj(des)

        semantic = self.semantic_expert(
            torch.cat([des_h, text_mean, amr_mean, text_mean * amr_mean], dim=1)
        )
        temporal_input = torch.cat(
            [text_h, amr_h, text_h * amr_h, (text_h - amr_h).abs()], dim=2
        )
        temporal_out, _ = self.temporal_gru(temporal_input)
        scores = self.temporal_score(temporal_out).squeeze(-1)
        temporal_weights = torch.softmax(scores, dim=1)
        temporal = (temporal_out * temporal_weights.unsqueeze(-1)).sum(dim=1)
        temporal = self.temporal_norm(temporal)

        graph_h = self.graph_expert(graph)
        style = self.style_expert(self._style_input(des, text_mean, amr_mean, num_prop, llm_features))
        raw_parts = [des, texts.mean(dim=1), amrs.mean(dim=1), graph]
        if self.num_prop_dim > 0:
            if num_prop is None:
                num_prop = torch.zeros(des.shape[0], self.num_prop_dim, device=des.device, dtype=des.dtype)
            raw_parts.append(num_prop)
        if self.llm_feature_dim > 0:
            if llm_features is None:
                llm_features = torch.zeros(des.shape[0], self.llm_feature_dim, device=des.device, dtype=des.dtype)
            raw_parts.append(llm_features)
        raw_h = self.raw_evidence(torch.cat(raw_parts, dim=1))
        experts = torch.stack([semantic, temporal, graph_h, style], dim=1)

        expert_logits = torch.stack(
            [head(experts[:, idx, :]) for idx, head in enumerate(self.expert_heads)], dim=1
        )
        expert_prob = torch.softmax(expert_logits.detach(), dim=-1)
        entropy = -(expert_prob * torch.log(expert_prob.clamp_min(1e-8))).sum(dim=-1)
        confidence = 1.0 - entropy / max(float(torch.log(torch.tensor(self.num_classes))), 1.0)
        router_input = torch.cat([experts.flatten(1), confidence], dim=1)
        route_logits = self.router(router_input)
        gate = torch.softmax(route_logits / max(self.route_temperature, 1e-4), dim=1)

        weighted = (experts * gate.unsqueeze(-1)).sum(dim=1)
        semantic_temporal = self.interaction_proj(torch.cat([semantic, temporal], dim=1))
        graph_style = self.interaction_proj(torch.cat([graph_h, style], dim=1))
        fused = self.fusion_norm(weighted + 0.15 * semantic_temporal + 0.15 * graph_style + 0.25 * raw_h)

        classifier_logits = self.classifier(fused)
        normalized_fused = F.normalize(fused, dim=1)
        normalized_prototypes = F.normalize(self.prototypes, dim=1)
        prototype_logits = normalized_fused @ normalized_prototypes.T / max(self.prototype_temperature, 1e-4)
        routed_expert_logits = (expert_logits * gate.unsqueeze(-1)).sum(dim=1)
        logits = classifier_logits + 0.15 * prototype_logits + 0.10 * routed_expert_logits + 0.25 * self.raw_classifier(raw_h)

        gate_mean = gate.mean(dim=0)
        route_balance = (gate_mean - (1.0 / 4.0)).pow(2).mean()
        route_confidence = F.mse_loss(gate, confidence / confidence.sum(dim=1, keepdim=True).clamp_min(1e-6))
        aux = {
            "prototype_logits": prototype_logits,
            "expert_logits": expert_logits,
            "route_balance": route_balance,
            "route_confidence": route_confidence,
            "temporal_weights": temporal_weights,
            "stream_weights": stream_weights,
        }
        return logits, gate, fused, aux


class PrototypeHead(nn.Module):
    """Class prototype regularizer used to separate the final feature space."""

    def __init__(self, embedding_dimension, num_classes, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.prototypes = nn.Parameter(torch.empty(num_classes, embedding_dimension))
        nn.init.xavier_uniform_(self.prototypes)

    def logits(self, embeddings):
        embeddings = F.normalize(embeddings, dim=1)
        prototypes = F.normalize(self.prototypes, dim=1)
        return torch.matmul(embeddings, prototypes.T) / max(self.temperature, 1e-6)

    def loss(self, embeddings, labels):
        if labels is None:
            return torch.tensor(0.0, device=embeddings.device)

        align_loss = F.cross_entropy(self.logits(embeddings), labels)
        prototypes = F.normalize(self.prototypes, dim=1)
        similarity = torch.matmul(prototypes, prototypes.T)
        off_diag = similarity - torch.eye(similarity.size(0), device=similarity.device)
        separation_loss = F.relu(off_diag - 0.05).pow(2).mean()
        return align_loss + 0.05 * separation_loss


class BotDMM(nn.Module):
    """BotDMM baseline used for comparison experiments."""

    MODE_ALIASES = {
        "botdmm": "base",
        "aemp": "base",
    }

    def __init__(
        self,
        des_size=768,
        tweet_size=768,
        amr_size=768,
        num_prop_size=13,
        llm_features_size=11,
        embedding_dimension=64,
        feature_dim=32,
        num_temporal_steps=5,
        dropout=0.2,
        temperature=0.07,
        alpha=0.5,
        num_heads=8,
        num_classes=3,
        ablation_mode="aemp",
        num_domains=3,
        grl_lambda=1.0,
        use_raep=False,
        use_hierarchy=False,
        use_hcrp=False,
        hcrp_route_mix=0.45,
        hcrp_hierarchy_mix=0.35,
        use_four_expert=False,
        four_expert_mix=0.15,
        freeze_backbone=False,
        use_behavior_expert=False,
    ):
        super().__init__()
        del num_heads
        ablation_mode = self.MODE_ALIASES.get(ablation_mode, ablation_mode)
        valid_modes = {
            "base",
            "no_memory",
            "no_event",
            "no_prototype",
            "no_domain",
            "no_structure",
            "no_content",
            "no_orthogonal",
        }
        if ablation_mode not in valid_modes:
            raise ValueError(f"Unsupported ablation_mode: {ablation_mode}")

        self.num_temporal_steps = num_temporal_steps
        self.embedding_dimension = embedding_dimension
        self.feature_dim = feature_dim
        self.temperature = temperature
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.grl_lambda = grl_lambda
        self.ablation_mode = ablation_mode
        self.use_raep = bool(use_raep)
        self.use_hierarchy = bool(use_hierarchy)
        self.use_hcrp = bool(use_hcrp)
        self.hcrp_route_mix = float(hcrp_route_mix)
        self.hcrp_hierarchy_mix = float(hcrp_hierarchy_mix)
        self.use_four_expert = bool(use_four_expert)
        self.four_expert_mix = float(four_expert_mix)
        self.freeze_backbone = bool(freeze_backbone)
        self.use_behavior_expert = bool(use_behavior_expert)

        self.des_encoder = nn.Linear(des_size, feature_dim)
        self.num_prop_encoder = nn.Linear(num_prop_size, feature_dim)
        self.llm_features_encoder = nn.Linear(llm_features_size, feature_dim)

        self.tweet_lstm = nn.LSTM(
            input_size=tweet_size,
            hidden_size=feature_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_temporal_steps > 1 else 0.0,
        )
        self.amr_lstm = nn.LSTM(
            input_size=amr_size,
            hidden_size=feature_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_temporal_steps > 1 else 0.0,
        )

        fused_dim = 5 * feature_dim
        self.content_decoupler = ContentDecoupler(fused_dim, embedding_dimension, dropout)
        self.structure_decoupler = StructureDecoupler(fused_dim, embedding_dimension, dropout)
        self.orthogonal_regularizer = OrthogonalConstraint(alpha=alpha)

        self.structure_projector = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.ReLU(),
            nn.Linear(embedding_dimension, embedding_dimension),
        )
        self.content_projector = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.ReLU(),
            nn.Linear(embedding_dimension, embedding_dimension),
        )
        self.class_projector = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.LayerNorm(embedding_dimension),
            nn.ReLU(),
            nn.Linear(embedding_dimension, embedding_dimension),
        )

        self.feature_integrator = nn.Sequential(
            nn.Linear(embedding_dimension * 2, embedding_dimension),
            nn.LayerNorm(embedding_dimension),
            nn.ReLU(),
        )
        self.temporal_pooling = TemporalAttention(embedding_dimension, dropout)
        self.prototype_head = PrototypeHead(embedding_dimension, num_classes, temperature)
        self.output_layer = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension // 2),
            nn.LayerNorm(embedding_dimension // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dimension // 2, num_classes),
        )
        # HCRP routes the final decision using separate content/structure
        # confidence estimates. The original BotDMM path is unchanged when
        # use_hcrp is disabled.
        self.content_classifier = nn.Sequential(
            nn.LayerNorm(embedding_dimension),
            nn.Dropout(dropout),
            nn.Linear(embedding_dimension, num_classes),
        )
        self.structure_classifier = nn.Sequential(
            nn.LayerNorm(embedding_dimension),
            nn.Dropout(dropout),
            nn.Linear(embedding_dimension, num_classes),
        )
        self.reliability_router = nn.Sequential(
            nn.Linear(4, max(16, embedding_dimension // 2)),
            nn.GELU(),
            nn.Linear(max(16, embedding_dimension // 2), 2),
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension // 2),
            nn.LayerNorm(embedding_dimension // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dimension // 2, num_domains),
        )
        # Reliability-aware graph purification. The edge gate is deliberately
        # initialized near an uninformative prior and acts as a soft weight on
        # the existing split-level graph rather than changing the dataset.
        self.edge_node_projector = nn.Linear(fused_dim, embedding_dimension)
        self.edge_reliability = nn.Sequential(
            nn.Linear(embedding_dimension * 4, embedding_dimension),
            nn.LayerNorm(embedding_dimension),
            nn.GELU(),
            nn.Linear(embedding_dimension, 1),
        )
        # Start from a neutral 0.5 reliability prior.  After the attention
        # renormalization this is exactly the original unweighted graph path,
        # so RAEP cannot destabilize BotDMM at initialization.
        nn.init.zeros_(self.edge_reliability[-1].weight)
        nn.init.zeros_(self.edge_reliability[-1].bias)
        # Four-expert residual fusion is function-preserving at initialization.
        self.metadata_expert = nn.Sequential(
            nn.Linear(feature_dim * 3, embedding_dimension),
            nn.LayerNorm(embedding_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dimension, embedding_dimension),
        )
        raw_metadata_dim = int(des_size + num_prop_size + llm_features_size)
        self.raw_metadata_expert = nn.Sequential(
            nn.Linear(raw_metadata_dim, embedding_dimension * 2),
            nn.LayerNorm(embedding_dimension * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dimension * 2, embedding_dimension),
        )
        self.four_expert_heads = nn.ModuleList(
            [nn.Linear(embedding_dimension, num_classes) for _ in range(4)]
        )
        self.four_router = nn.Sequential(
            nn.Linear(embedding_dimension * 4 + 4, embedding_dimension),
            nn.LayerNorm(embedding_dimension),
            nn.GELU(),
            nn.Linear(embedding_dimension, 4),
        )
        self.four_residual = nn.Sequential(
            nn.LayerNorm(embedding_dimension * 4),
            nn.Linear(embedding_dimension * 4, embedding_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dimension, num_classes),
        )
        # Dedicated Human/Traditional boundary head.  It sees the same four
        # expert representations used by the router, keeping the residual
        # parameterized by the paper's four-expert evidence rather than by a
        # second, much larger boundary network.
        self.pairwise_head = nn.Sequential(
            nn.LayerNorm(embedding_dimension * 4),
            nn.Linear(embedding_dimension * 4, embedding_dimension),
            nn.GELU(),
            nn.Linear(embedding_dimension, 2),
        )
        # The released data contain counterfactual variants of the same base
        # profile under different automation policies.  This fourth expert
        # therefore uses temporal/content-structure changes, not static
        # profile metadata that is intentionally identical across variants.
        # It is declared after the legacy modules so adding this variant does
        # not perturb the random initialization of the verified baseline.
        self.behavior_expert = nn.Sequential(
            nn.Linear(embedding_dimension * 6, embedding_dimension * 2),
            nn.LayerNorm(embedding_dimension * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dimension * 2, embedding_dimension),
        )
        self.four_residual_scale = nn.Parameter(torch.zeros(1))
        # Hierarchical supervision: human vs. automated, then automated type.
        self.automation_classifier = nn.Linear(embedding_dimension, 2)
        self.bot_type_classifier = nn.Linear(embedding_dimension, max(num_classes - 1, 2))

    def forward(
        self,
        des,
        tweets,
        amrs,
        num_prop,
        llm_features,
        edge_indices_list,
        labels=None,
        source_labels=None,
    ):
        des_feat = self.des_encoder(des)
        num_feat = self.num_prop_encoder(num_prop)
        llm_feat = self.llm_features_encoder(llm_features)

        tweets_tensor = torch.stack(tweets, dim=1)
        tweet_lstm_out, _ = self.tweet_lstm(tweets_tensor)
        amrs_tensor = torch.stack(amrs, dim=1)
        amr_lstm_out, _ = self.amr_lstm(amrs_tensor)

        temporal_embeddings = []
        content_seq = []
        structure_seq = []
        edge_reliability_means = []
        steps_to_use = min(self.num_temporal_steps, len(tweets))

        for step in range(steps_to_use):
            tweet_feat = tweet_lstm_out[:, step, :]
            amr_feat = amr_lstm_out[:, step, :]
            combined_features = torch.cat(
                [des_feat, tweet_feat, amr_feat, num_feat, llm_feat],
                dim=1,
            )

            content_features = self.content_decoupler(combined_features)
            edge_index = edge_indices_list[step]
            if edge_index.size(1) > 0:
                valid_mask = (
                    (edge_index[0] < combined_features.size(0))
                    & (edge_index[1] < combined_features.size(0))
                )
                edge_index = edge_index[:, valid_mask]

            edge_weight = None
            if self.use_raep and edge_index.size(1) > 0:
                edge_nodes = self.edge_node_projector(combined_features)
                src, dst = edge_index
                src_h, dst_h = edge_nodes[src], edge_nodes[dst]
                edge_pair = torch.cat(
                    [src_h, dst_h, (src_h - dst_h).abs(), src_h * dst_h], dim=1
                )
                edge_weight = torch.sigmoid(self.edge_reliability(edge_pair)).squeeze(-1)
                edge_reliability_means.append(edge_weight.mean())
            else:
                edge_reliability_means.append(torch.tensor(0.0, device=combined_features.device))

            structure_features = self.structure_decoupler(
                combined_features, edge_index, edge_weight=edge_weight
            )
            if self.ablation_mode == "no_structure":
                structure_features = torch.zeros_like(content_features)
            elif self.ablation_mode == "no_content":
                content_features = torch.zeros_like(structure_features)

            if self.ablation_mode == "no_orthogonal":
                structure_features_norm = F.normalize(structure_features, dim=1)
                content_features_norm = F.normalize(content_features, dim=1)
            else:
                structure_features_norm, content_features_norm = self.orthogonal_regularizer(
                    structure_features,
                    content_features,
                )

            structure_proj = F.normalize(self.structure_projector(structure_features), dim=1)
            content_proj = F.normalize(self.content_projector(content_features), dim=1)
            node_feat = torch.cat([structure_features_norm, content_features_norm], dim=1)
            node_feat = self.feature_integrator(node_feat)

            temporal_embeddings.append(node_feat)
            structure_seq.append(structure_proj)
            content_seq.append(content_proj)

        if not temporal_embeddings:
            raise ValueError("QuadFusion-Net requires at least one temporal text/AMR step.")

        while len(temporal_embeddings) < self.num_temporal_steps:
            temporal_embeddings.append(temporal_embeddings[-1])

        temporal_sequence = torch.stack(temporal_embeddings, dim=1)
        final_embedding = self.temporal_pooling(temporal_sequence)
        if self.freeze_backbone:
            des_feat = des_feat.detach()
            num_feat = num_feat.detach()
            llm_feat = llm_feat.detach()
            final_embedding = final_embedding.detach()
        view_gate_weights = []
        base_logits = self.output_layer(final_embedding)

        # Route between independent content and structure decisions using
        # predictive confidence and cross-view agreement.
        content_stack = torch.stack(content_seq, dim=1)
        structure_stack = torch.stack(structure_seq, dim=1)
        if self.freeze_backbone:
            content_stack = content_stack.detach()
            structure_stack = structure_stack.detach()
        four_expert_loss = torch.tensor(0.0, device=final_embedding.device)
        four_gate = None
        four_residual_alpha = torch.tensor(0.0, device=final_embedding.device)
        pairwise_loss = torch.tensor(0.0, device=final_embedding.device)
        if self.use_four_expert:
            content_embedding = F.normalize(content_stack.mean(dim=1), dim=1)
            structure_embedding = F.normalize(structure_stack.mean(dim=1), dim=1)
            temporal_embedding = F.normalize(final_embedding, dim=1)
            if self.use_behavior_expert:
                content_delta = (content_stack[:, 1:] - content_stack[:, :-1]).abs().mean(dim=1)
                structure_delta = (structure_stack[:, 1:] - structure_stack[:, :-1]).abs().mean(dim=1)
                cross_view_delta = (content_embedding - structure_embedding).abs()
                behavior_input = torch.cat(
                    [content_embedding, structure_embedding, temporal_embedding,
                     content_delta, structure_delta, cross_view_delta], dim=1
                )
                metadata_embedding = F.normalize(self.behavior_expert(behavior_input), dim=1)
            else:
                metadata_embedding = F.normalize(
                    self.raw_metadata_expert(torch.cat([des, num_prop, llm_features], dim=1)),
                    dim=1,
                )
            four_embeddings = torch.stack(
                [content_embedding, structure_embedding, temporal_embedding, metadata_embedding],
                dim=1,
            )
            four_logits = torch.stack(
                [head(four_embeddings[:, idx, :]) for idx, head in enumerate(self.four_expert_heads)],
                dim=1,
            )
            expert_prob = F.softmax(four_logits.detach(), dim=-1)
            entropy = -(expert_prob * torch.log(expert_prob.clamp_min(1e-8))).sum(dim=-1)
            entropy_norm = torch.log(torch.tensor(float(self.num_classes), device=final_embedding.device)).clamp_min(1.0)
            confidence = 1.0 - entropy / entropy_norm
            router_input = torch.cat([four_embeddings.flatten(1), confidence], dim=1)
            four_gate = torch.softmax(self.four_router(router_input), dim=1)
            routed_four_logits = (four_logits * four_gate.unsqueeze(-1)).sum(dim=1)
            residual_logits = self.four_residual(four_embeddings.flatten(1))
            pairwise_logits = self.pairwise_head(four_embeddings.flatten(1))
            four_residual_alpha = self.four_expert_mix * torch.tanh(self.four_residual_scale)
            # Interpolate with the trusted BotDMM decision surface instead of
            # replacing it with an unconstrained auxiliary prediction.
            # Human/Traditional Bot is the shared hard boundary in both
            # releases.  Compare centered two-class logits so the pairwise
            # head contributes a boundary direction rather than an arbitrary
            # offset whose scale is unrelated to the BotDMM head.
            pair_centered = pairwise_logits - pairwise_logits.mean(dim=1, keepdim=True)
            base_pair_centered = base_logits[:, :2] - base_logits[:, :2].mean(dim=1, keepdim=True)
            pair_delta = torch.zeros_like(base_logits)
            pair_delta[:, :2] = pair_centered - base_pair_centered
            expert_delta = 0.45 * (routed_four_logits - base_logits) + 0.15 * residual_logits + 0.40 * pair_delta
            logits = base_logits + four_residual_alpha * expert_delta
            if labels is not None:
                expert_losses = [F.cross_entropy(four_logits[:, idx, :], labels) for idx in range(four_logits.size(1))]
                pair_mask = labels < 2
                if pair_mask.any():
                    pairwise_loss = F.cross_entropy(pairwise_logits[pair_mask], labels[pair_mask])
                four_expert_loss = (
                    0.25 * torch.stack(expert_losses).mean()
                    + 0.25 * F.cross_entropy(routed_four_logits, labels)
                    + 0.50 * pairwise_loss
                )

        content_embedding = content_stack.mean(dim=1)
        structure_embedding = structure_stack.mean(dim=1)
        content_logits = self.content_classifier(content_embedding)
        structure_logits = self.structure_classifier(structure_embedding)
        content_prob = F.softmax(content_logits, dim=-1)
        structure_prob = F.softmax(structure_logits, dim=-1)
        entropy_norm = torch.log(torch.tensor(float(self.num_classes), device=final_embedding.device)).clamp_min(1.0)
        content_conf = 1.0 - (-(content_prob * torch.log(content_prob.clamp_min(1e-8))).sum(dim=1) / entropy_norm)
        structure_conf = 1.0 - (-(structure_prob * torch.log(structure_prob.clamp_min(1e-8))).sum(dim=1) / entropy_norm)
        agreement = F.cosine_similarity(content_embedding, structure_embedding, dim=1).add(1.0).mul(0.5).clamp(0.0, 1.0)
        route_features = torch.stack(
            [content_conf, structure_conf, agreement, (content_conf - structure_conf).abs()],
            dim=1,
        )
        route_weights = torch.softmax(self.reliability_router(route_features), dim=1)
        routed_logits = (
            route_weights[:, 0:1] * content_logits
            + route_weights[:, 1:2] * structure_logits
        )

        if self.use_hcrp:
            flat_prob = F.softmax(base_logits, dim=-1)
            routed_prob = F.softmax(routed_logits, dim=-1)
            mixed_prob = (
                (1.0 - self.hcrp_route_mix) * flat_prob
                + self.hcrp_route_mix * routed_prob
            )
            logits = torch.log(mixed_prob.clamp_min(1e-8))
        elif not self.use_four_expert:
            logits = base_logits

        automation_logits = self.automation_classifier(final_embedding) if self.use_hierarchy else None
        bot_type_logits = self.bot_type_classifier(final_embedding) if self.use_hierarchy else None
        hierarchy_prob = None
        hierarchy_consistency_loss = torch.tensor(0.0, device=final_embedding.device)
        if self.use_hierarchy:
            coarse_prob = F.softmax(automation_logits, dim=-1)
            fine_prob = F.softmax(bot_type_logits, dim=-1)
            hierarchy_prob = torch.cat(
                [coarse_prob[:, 0:1], coarse_prob[:, 1:2] * fine_prob],
                dim=1,
            )
            flat_prob = F.softmax(logits, dim=-1)
            hierarchy_consistency_loss = 0.5 * (
                F.kl_div(torch.log(flat_prob.clamp_min(1e-8)), hierarchy_prob, reduction="batchmean")
                + F.kl_div(torch.log(hierarchy_prob.clamp_min(1e-8)), flat_prob, reduction="batchmean")
            )
            if self.use_hcrp:
                final_prob = (
                    (1.0 - self.hcrp_hierarchy_mix) * flat_prob
                    + self.hcrp_hierarchy_mix * hierarchy_prob
                )
                logits = torch.log(final_prob.clamp_min(1e-8))

        if self.freeze_backbone:
            structure_proj = structure_proj.detach()
            content_proj = content_proj.detach()

        if self.ablation_mode in {"no_structure", "no_content"}:
            feature_contrastive_loss = torch.tensor(0.0, device=final_embedding.device)
        else:
            feature_contrastive_loss = self.feature_contrastive_loss(structure_proj, content_proj)

        if labels is not None:
            projected_embedding = self.class_projector(final_embedding)
            class_contrastive_loss = self.class_contrastive_loss(projected_embedding, labels)
            prototype_loss = (
                torch.tensor(0.0, device=final_embedding.device)
                if self.ablation_mode == "no_prototype"
                else self.prototype_head.loss(projected_embedding, labels)
            )
        else:
            projected_embedding = self.class_projector(final_embedding)
            class_contrastive_loss = torch.tensor(0.0, device=final_embedding.device)
            prototype_loss = torch.tensor(0.0, device=final_embedding.device)

        domain_loss = self.domain_adversarial_loss(final_embedding, source_labels)

        return {
            "logits": logits,
            "final_embedding": final_embedding,
            "structure_features": structure_features,
            "content_features": content_features,
            "structure_proj": structure_proj,
            "content_proj": content_proj,
            "feature_contrastive_loss": feature_contrastive_loss,
            "class_contrastive_loss": class_contrastive_loss,
            "prototype_loss": prototype_loss,
            "domain_loss": domain_loss,
            "automation_logits": automation_logits,
            "bot_type_logits": bot_type_logits,
            "hierarchy_prob": hierarchy_prob,
            "hierarchy_consistency_loss": hierarchy_consistency_loss,
            "content_logits": content_logits,
            "structure_logits": structure_logits,
            "reliability_weights": route_weights,
            "content_confidence": content_conf,
            "structure_confidence": structure_conf,
            "edge_reliability_mean": torch.stack(edge_reliability_means).mean(),
            "structure_seq": structure_seq,
            "content_seq": content_seq,
            "view_gate_weights": view_gate_weights,
            "four_expert_loss": four_expert_loss,
            "four_gate": four_gate,
            "four_residual_alpha": four_residual_alpha,
            "pairwise_loss": pairwise_loss,
        }

    def feature_contrastive_loss(self, structure_proj, content_proj):
        batch_size = structure_proj.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=structure_proj.device)

        structure_proj = F.normalize(structure_proj, dim=1)
        content_proj = F.normalize(content_proj, dim=1)
        logits = torch.matmul(structure_proj, content_proj.T) / self.temperature
        logits = torch.clamp(logits, -50.0, 50.0)
        targets = torch.arange(batch_size, device=structure_proj.device)
        loss = 0.5 * (
            F.cross_entropy(logits, targets)
            + F.cross_entropy(logits.T, targets)
        )
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=structure_proj.device)
        return loss

    def class_contrastive_loss(self, embeddings, labels):
        batch_size = embeddings.shape[0]
        if labels is None or batch_size < 4:
            return torch.tensor(0.0, device=embeddings.device)

        embeddings_normalized = F.normalize(embeddings, p=2, dim=1)
        similarity_matrix = torch.matmul(embeddings_normalized, embeddings_normalized.T)
        similarity_matrix = torch.clamp(similarity_matrix / self.temperature, -50.0, 50.0)

        labels_matrix = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
        mask_self = torch.eye(batch_size, device=embeddings.device)
        labels_matrix = labels_matrix * (1.0 - mask_self)

        has_positives = labels_matrix.sum(dim=1) > 0
        if not has_positives.any():
            return torch.tensor(0.0, device=embeddings.device)

        valid_indices = torch.where(has_positives)[0]
        exp_sim = torch.exp(similarity_matrix)
        pos_sim = torch.zeros(batch_size, device=embeddings.device)
        denom = torch.zeros(batch_size, device=embeddings.device)

        for idx in valid_indices:
            pos_mask = labels_matrix[idx]
            neg_mask = (1.0 - labels_matrix[idx]) * (1.0 - mask_self[idx])
            pos_sim[idx] = (exp_sim[idx] * pos_mask).sum()
            denom[idx] = (exp_sim[idx] * (pos_mask + neg_mask)).sum()

        valid_pos_sim = pos_sim[valid_indices]
        valid_denom = denom[valid_indices] + 1e-8
        loss = -torch.log(valid_pos_sim / valid_denom + 1e-8).mean()
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=embeddings.device)
        return loss

    def domain_adversarial_loss(self, embeddings, source_labels):
        if (
            source_labels is None
            or self.ablation_mode == "no_domain"
            or self.num_domains <= 1
        ):
            return torch.tensor(0.0, device=embeddings.device)

        valid = (source_labels >= 0) & (source_labels < self.num_domains)
        if valid.sum() < 2:
            return torch.tensor(0.0, device=embeddings.device)

        reversed_features = grad_reverse(embeddings[valid], self.grl_lambda)
        domain_logits = self.domain_classifier(reversed_features)
        return F.cross_entropy(domain_logits, source_labels[valid])

