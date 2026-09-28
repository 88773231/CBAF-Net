import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score
import logging

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, model, config, device='cuda'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        class_weights = config.get('class_weights')
        if class_weights is not None:
            class_weights = class_weights.to(device)
        self.class_weights = class_weights
        self.label_smoothing = config.get('label_smoothing', 0.0)
        self.focal_gamma = config.get('focal_gamma', 0.0)
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=self.label_smoothing
        )
        self.aux_criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.get('lr', 0.00005),
            weight_decay=config.get('weight_decay', 5e-4)
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        self.save_dir = config.get('save_dir', './models_saved')
        os.makedirs(self.save_dir, exist_ok=True)

    def _classification_loss(self, logits, labels):
        if self.focal_gamma <= 0:
            return self.criterion(logits, labels)

        ce_loss = F.cross_entropy(
            logits,
            labels,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
            reduction='none',
        )
        pt = torch.exp(-ce_loss.detach())
        focal_weight = (1.0 - pt).pow(self.focal_gamma)
        return (focal_weight * ce_loss).mean()

    def _unpack_batch(self, batch):
        if len(batch) == 8:
            des, tweets, amrs, num_prop, llm_features, edge_indices, labels, source_labels = batch
        else:
            des, tweets, amrs, num_prop, llm_features, edge_indices, labels = batch
            source_labels = torch.full_like(labels, -1)
        des = des.to(self.device)
        num_prop = num_prop.to(self.device)
        llm_features = llm_features.to(self.device)
        labels = labels.to(self.device)
        source_labels = source_labels.to(self.device)
        tweets = [t.to(self.device) for t in tweets]
        amrs = [a.to(self.device) for a in amrs]
        edge_indices = [e.to(self.device) for e in edge_indices]
        return des, tweets, amrs, num_prop, llm_features, edge_indices, labels, source_labels

    def train_epoch(self, train_loader):
        if self.config.get('freeze_backbone', False):
            self.model.eval()
            for name in ('metadata_expert', 'raw_metadata_expert', 'four_expert_heads', 'four_router', 'four_residual', 'pairwise_head'):
                module = getattr(self.model, name, None)
                if module is not None:
                    module.train()
        else:
            self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        for batch in tqdm(train_loader, desc="Training"):
            des, tweets, amrs, num_prop, llm_features, edge_indices, labels, source_labels = self._unpack_batch(batch)
            self.optimizer.zero_grad()
            outputs = self.model(
                des,
                tweets,
                amrs,
                num_prop,
                llm_features,
                edge_indices,
                labels=labels,
                source_labels=source_labels,
            )

            cls_loss = self._classification_loss(outputs['logits'], labels)
            cont_loss = outputs['feature_contrastive_loss']
            class_cont_loss = outputs['class_contrastive_loss']
            prototype_loss = outputs.get('prototype_loss', torch.tensor(0.0, device=self.device))
            domain_loss = outputs.get('domain_loss', torch.tensor(0.0, device=self.device))
            moe_balance_loss = outputs.get('moe_balance_loss', torch.tensor(0.0, device=self.device))
            consistency_loss = outputs.get('hierarchy_consistency_loss', torch.tensor(0.0, device=self.device))
            four_expert_loss = outputs.get('four_expert_loss', torch.tensor(0.0, device=self.device))
            hierarchy_loss = torch.tensor(0.0, device=self.device)
            if outputs.get('automation_logits') is not None:
                automation_labels = (labels > 0).long()
                hierarchy_loss = hierarchy_loss + self.aux_criterion(
                    outputs['automation_logits'],
                    automation_labels,
                )
                bot_mask = labels > 0
                if bot_mask.any():
                    hierarchy_loss = hierarchy_loss + self.aux_criterion(
                        outputs['bot_type_logits'][bot_mask],
                        labels[bot_mask] - 1,
                    )

            # ``base`` remains the exact BotDMM objective.  The enhanced
            # variant keeps the same trunk but opts into the auxiliary
            # contrastive and hierarchical terms explicitly.
            if self.config.get('ablation_mode') == 'base' and not self.config.get('enhanced', False):
                total_loss_batch = cls_loss
            else:
                total_loss_batch = (
                    cls_loss +
                    self.config.get('lambda_feature_contrast', 0.1) * cont_loss +
                    self.config.get('lambda_class_contrast', 0.1) * class_cont_loss +
                    self.config.get('lambda_prototype', 0.0) * prototype_loss +
                    self.config.get('lambda_domain', 0.0) * domain_loss +
                    self.config.get('lambda_hierarchy', 0.0) * hierarchy_loss +
                    self.config.get('lambda_consistency', 0.0) * consistency_loss +
                    self.config.get('lambda_moe_balance', 0.0) * moe_balance_loss +
                    self.config.get('lambda_four_expert', 0.0) * four_expert_loss
                )

            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get('clip_grad', 1.0))
            self.optimizer.step()

            total_loss += total_loss_batch.item()
            preds = outputs['logits'].argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(train_loader)
        train_acc = accuracy_score(all_labels, all_preds)
        return avg_loss, train_acc

    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                des, tweets, amrs, num_prop, llm_features, edge_indices, labels, _ = self._unpack_batch(batch)
                outputs = self.model(des, tweets, amrs, num_prop, llm_features, edge_indices)
                loss = self._classification_loss(outputs['logits'], labels)
                total_loss += loss.item()
                preds = outputs['logits'].argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(val_loader)
        num_classes = self.config.get('num_classes', 3)
        avg_method = 'binary' if num_classes == 2 else 'macro'

        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1': f1_score(all_labels, all_preds, average=avg_method),
            'mcc': matthews_corrcoef(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, average=avg_method, zero_division=0),
            'recall': recall_score(all_labels, all_preds, average=avg_method, zero_division=0)
        }
        return metrics

    def train(self, train_loader, val_loader, num_epochs):
        val_metric = self.config.get('val_metric', 'accuracy')
        if val_metric not in {'accuracy', 'f1', 'mcc'}:
            raise ValueError(f"Unsupported val_metric: {val_metric}")

        best_val_score = float('-inf')
        patience_counter = 0
        best_model_path = os.path.join(self.save_dir, 'best_model.pt')

        if self.config.get('include_initial_checkpoint', False):
            initial_metrics = self.evaluate(val_loader)
            best_val_score = initial_metrics[val_metric]
            self.save_checkpoint(-1, initial_metrics, best_model_path, metric_name=val_metric)
            logger.info(
                f"Initial checkpoint - Val Loss: {initial_metrics['loss']:.4f}, "
                f"Val Acc: {initial_metrics['accuracy']:.4f}, "
                f"Val F1: {initial_metrics['f1']:.4f}, Val MCC: {initial_metrics['mcc']:.4f}"
            )
            if self.config.get('use_four_expert', False) and hasattr(self.model, 'four_residual_scale'):
                warmup = self.config.get('four_warmup_scale')
                if warmup is not None:
                    with torch.no_grad():
                        self.model.four_residual_scale.fill_(float(warmup))
                    logger.info('Four-expert residual warm-up scale set to %.4f after baseline preservation.', float(warmup))

        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            train_loss, train_acc = self.train_epoch(train_loader)
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

            val_metrics = self.evaluate(val_loader)
            logger.info(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
            logger.info(f"Val F1: {val_metrics['f1']:.4f}, Val MCC: {val_metrics['mcc']:.4f}")

            self.scheduler.step(val_metrics['accuracy'])
            current_score = val_metrics[val_metric]

            if current_score > best_val_score:
                best_val_score = current_score
                patience_counter = 0
                self.save_checkpoint(epoch, val_metrics, best_model_path, metric_name=val_metric)
            else:
                patience_counter += 1

            if patience_counter >= self.config.get('patience', 20):
                logger.info("Early stopping triggered")
                break

        return best_val_score, best_model_path

    def save_checkpoint(self, epoch, metrics, path=None, metric_name='accuracy'):
        if path is None:
            path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(self.model.state_dict(), path)
        logger.info(
            f"Model saved to {path} "
            f"(Val {metric_name}: {metrics[metric_name]:.4f}, Val Acc: {metrics['accuracy']:.4f})"
        )

