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
    """Legacy multi-expert model retained for import compatibility.

    The formal CBAF-Net evaluator below uses the frozen ``BotDMM`` baseline
    and an independent behavioral-statistics expert; this class is not part of
    the reported decision-level fusion path.
    """

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
        # Historical training/evaluation configs used ``aemp`` for the
        # baseline path.  Keep the alias so constructing BotDMM() with its
        # legacy default remains valid while the released evaluator passes
        # the explicit ``base`` mode.
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
        self.domain_classifier = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension // 2),
            nn.LayerNorm(embedding_dimension // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dimension // 2, num_domains),
        )

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

            structure_features = self.structure_decoupler(combined_features, edge_index)
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
        view_gate_weights = []
        logits = self.output_layer(final_embedding)

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
            "structure_seq": structure_seq,
            "content_seq": content_seq,
            "view_gate_weights": view_gate_weights,
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
