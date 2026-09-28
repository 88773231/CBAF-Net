import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if not os.environ.get("OMP_NUM_THREADS"):
    os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import json
import subprocess
import torch
import numpy as np
import logging
from pathlib import Path
from torch.utils.data import DataLoader

from training.trainer import Trainer
from models.botdmm import BotDMM
from data.data_loader import Twibot20Dataset, BotSimDataset, QuadBotDataset, collate_fn

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


MODEL_VARIANT_TO_MODE = {
    "base": "base",
    "botdmm": "base",
    "aemp": "aemp",
    "quadfusion": "aemp",
    "cbaf": "aemp",
    "cbaf_net": "aemp",
    # RAEP + CHDF keeps the BotDMM trunk and activates only the two proposed
    # additions.  The trainer uses ``enhanced`` to enable auxiliary losses.
    "raep_chdf": "base",
    "raep_only": "base",
    "chdf_only": "base",
    "hcrp": "base",
    "four_expert": "base",
    "four_expert_behavior": "base",
    "four_expert_tabular": "base",
    "four_expert_raep": "base",
}

MODEL_DISPLAY_NAMES = {
    "cbaf": "CBAF-Net",
    "cbaf_net": "CBAF-Net",
    "aemp": "QuadFusion-Net (legacy alias)",
    "quadfusion": "QuadFusion-Net (legacy alias)",
    "botdmm": "BotDMM",
    "base": "BotDMM",
    "raep_chdf": "BotDMM-RAEP-CHDF",
    "raep_only": "BotDMM+RAEP",
    "chdf_only": "BotDMM+CHDF",
    "hcrp": "BotDMM-HCRP",
    "four_expert": "BotDMM-FourExpert-Residual",
    "four_expert_behavior": "BotDMM-FourExpert-Behavior",
    "four_expert_tabular": "BotDMM-FourExpert-Tabular",
    "four_expert_raep": "BotDMM-FourExpert-RAEP",
}

LEGACY_MODEL_VARIANTS = {"aemp", "quadfusion"}


def canonical_dataset_name(dataset):
    """Return the paper-facing dataset ID while accepting old CLI aliases."""
    value = str(dataset).strip().lower()
    if value in {"twibot20", "twibot22"}:
        return "twibot22"
    if value in {"botsim", "quadbot"}:
        return value
    raise ValueError(f"Unsupported dataset: {dataset}")


def default_if_none(value, fallback):
    return fallback if value is None else value


def uses_paper_cbaf(args):
    ablation_mode = MODEL_VARIANT_TO_MODE.get(args.model_variant, args.model_variant)
    return args.dataset in {"quadbot", "twibot22"} and ablation_mode == "aemp"


def find_formal_cbaf_script():
    """Locate the canonical hybrid evaluator used by the final release."""
    here = Path(__file__).resolve()
    candidates = (
        here.with_name("train_hybrid_bsedmm.py"),
        here.parent / "scripts" / "train_hybrid_bsedmm.py",
        here.parent.parent / "scripts" / "train_hybrid_bsedmm.py",
    )
    return next((path for path in candidates if path.is_file()), None)


def run_paper_cbaf(args):
    """Run the formal CBAF-Net hybrid evaluator and preserve old aliases."""
    if args.out:
        out_path = Path(args.out)
    elif args.model_variant in LEGACY_MODEL_VARIANTS:
        out_path = Path("results") / "quadfusion_single" / f"quadfusion_seed{args.seed}.json"
    else:
        out_path = Path("results") / "cbaf_single" / f"cbaf_seed{args.seed}.json"

    formal_script = find_formal_cbaf_script()
    if formal_script is not None and args.model_variant not in LEGACY_MODEL_VARIANTS:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(formal_script),
            "--dataset", args.dataset,
            "--data_dir", args.data_dir,
            "--seed", str(args.seed),
            "--out_dir", str(out_path.parent),
        ]
        if args.no_cuda:
            cmd.append("--no_cuda")
        logger.info("Running canonical CBAF-Net hybrid evaluator: %s", " ".join(cmd))
        subprocess.run(cmd, check=True)
        produced = out_path.parent / "metrics.json"
        if not produced.is_file():
            raise FileNotFoundError(f"CBAF-Net evaluator completed without {produced}")
        payload = json.loads(produced.read_text(encoding="utf-8"))
        payload["method"] = "CBAF-Net"
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        logger.info("Result JSON: %s", out_path.as_posix())
        return

    # Compatibility fallback for older remote containers that still ship the
    # historical backend.  The formal release path above is preferred.
    script_path = Path(__file__).with_name("train_quadfusion.py")
    if not script_path.is_file():
        raise FileNotFoundError(
            "No canonical scripts/train_hybrid_bsedmm.py was found and the "
            "legacy train_quadfusion.py backend is unavailable."
        )
    save_dir = Path(args.save_dir)

    cmd = [
        sys.executable,
        str(script_path),
        "--data_dir",
        args.data_dir,
        "--label_mode",
        # The backend still exposes its historical label-mode token; the
        # public CLI and result metadata use the canonical Twibot22 name.
        "quadbot" if args.dataset == "quadbot" else "twibot20",
        "--out",
        out_path.as_posix(),
        "--save_dir",
        save_dir.as_posix(),
        "--num_steps",
        str(args.num_steps),
        "--hidden_dim",
        str(args.cbaf_hidden_dim),
        "--dropout",
        str(default_if_none(args.dropout, 0.25)),
        "--lr",
        str(default_if_none(args.lr, 0.0008)),
        "--weight_decay",
        str(default_if_none(args.weight_decay, 0.0001)),
        "--epochs",
        str(default_if_none(args.epochs, 60)),
        "--patience",
        str(default_if_none(args.patience, 10)),
        "--batch_size",
        str(default_if_none(args.batch_size, 64)),
        "--label_smoothing",
        str(default_if_none(args.label_smoothing, 0.03)),
        "--seed",
        str(args.seed),
        "--gate_entropy_weight",
        str(args.gate_entropy_weight),
        "--contrastive_weight",
        str(args.contrastive_weight),
        "--contrastive_temperature",
        str(args.contrastive_temperature),
        "--expert_dropout",
        str(args.cbaf_expert_dropout),
        "--aux_weight",
        str(args.cbaf_aux_weight),
        "--ablation",
        args.cbaf_ablation,
    ]
    if args.save_embeddings:
        cmd.extend(["--save_embeddings", args.save_embeddings])
    if args.no_cuda:
        cmd.append("--no_cuda")

    if args.model_variant in LEGACY_MODEL_VARIANTS:
        logger.warning("Routing legacy QuadFusion alias; use --model_variant cbaf for formal runs.")
    else:
        logger.info("Routing legacy-compatible CBAF-Net backend.")
    logger.info("Command: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    if args.model_variant not in LEGACY_MODEL_VARIANTS and out_path.exists():
        try:
            payload = json.loads(out_path.read_text(encoding="utf-8"))
            changed = False
            for key in ("model", "baseline", "method"):
                if payload.get(key) in {"QuadFusion", "QuadFusion-Net", "QuadFusion-Net++"}:
                    payload[key] = "CBAF-Net"
                    changed = True
            if changed:
                out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not canonicalize formal result metadata at %s: %s", out_path, exc)
    logger.info("Result JSON: %s", out_path.as_posix())


def uses_paper_quadfusion(args):
    """Legacy function alias; formal callers should use :func:`uses_paper_cbaf`."""
    return uses_paper_cbaf(args)


def run_paper_quadfusion(args):
    """Legacy function alias; formal callers should use :func:`run_paper_cbaf`."""
    return run_paper_cbaf(args)


def train(args, config, device):
    logger.info("=" * 60)
    display_name = MODEL_DISPLAY_NAMES.get(args.model_variant, args.model_variant)
    logger.info(f"Training {display_name} - Dataset: {args.dataset}")
    logger.info("=" * 60)

    logger.info("Loading datasets...")
    if args.dataset == 'botsim':
        train_dataset = BotSimDataset(args.data_dir, split='train', num_steps=config['num_steps'])
        val_dataset = BotSimDataset(args.data_dir, split='val', num_steps=config['num_steps'])
        test_dataset = BotSimDataset(args.data_dir, split='test', num_steps=config['num_steps'])
        num_classes = 2
    elif args.dataset == 'quadbot':
        train_dataset = QuadBotDataset(args.data_dir, split='train', num_steps=config['num_steps'], preserve_tabular_scale=config.get('raw_tabular', False))
        val_dataset = QuadBotDataset(args.data_dir, split='val', num_steps=config['num_steps'], preserve_tabular_scale=config.get('raw_tabular', False))
        test_dataset = QuadBotDataset(args.data_dir, split='test', num_steps=config['num_steps'], preserve_tabular_scale=config.get('raw_tabular', False))
        num_classes = 4
    else:
        train_dataset = Twibot20Dataset(args.data_dir, split='train', num_steps=config['num_steps'], preserve_tabular_scale=config.get('raw_tabular', False))
        val_dataset = Twibot20Dataset(args.data_dir, split='val', num_steps=config['num_steps'], preserve_tabular_scale=config.get('raw_tabular', False))
        test_dataset = Twibot20Dataset(args.data_dir, split='test', num_steps=config['num_steps'], preserve_tabular_scale=config.get('raw_tabular', False))
        num_classes = 3

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")

    if args.class_weight == "balanced":
        counts = torch.bincount(train_dataset.labels.long(), minlength=num_classes).float()
        weights = counts.sum() / (counts.clamp_min(1.0) * num_classes)
        config['class_weights'] = weights
        logger.info("Class counts: %s", counts.tolist())
        logger.info("Class weights: %s", weights.tolist())
    elif args.class_weight == "traditional_up":
        # The formal releases have a persistent Human/Traditional confusion.
        # A small, explicit class-1 cost increase is used only as a training
        # objective; validation still selects the checkpoint by macro-F1.
        weights = torch.ones(num_classes, dtype=torch.float32)
        if num_classes > 1:
            weights[1] = 1.30
        config['class_weights'] = weights
        logger.info("Traditional-up class weights: %s", weights.tolist())
    else:
        config['class_weights'] = None

    sample = train_dataset[0]
    num_prop_size = sample['num_prop'].shape[0]
    llm_features_size = sample['llm_features'].shape[0]
    logger.info(f"num_prop_size: {num_prop_size}, llm_features_size: {llm_features_size}")

    model = BotDMM(
        des_size=768,
        tweet_size=768,
        amr_size=768,
        num_prop_size=num_prop_size,
        llm_features_size=llm_features_size,
        embedding_dimension=config['embedding_dim'],
        feature_dim=config['feature_dim'],
        num_temporal_steps=config['num_steps'],
        dropout=config['dropout'],
        temperature=config['temperature'],
        alpha=config['alpha'],
        num_classes=num_classes,
        ablation_mode=config['ablation_mode'],
        grl_lambda=config['grl_lambda'],
        use_raep=config.get('use_raep', False),
        use_hierarchy=config.get('use_hierarchy', False),
        use_hcrp=config.get('use_hcrp', False),
        hcrp_route_mix=config.get('hcrp_route_mix', 0.45),
        hcrp_hierarchy_mix=config.get('hcrp_hierarchy_mix', 0.35),
        use_four_expert=config.get('use_four_expert', False),
        four_expert_mix=config.get('four_expert_mix', 0.15),
        freeze_backbone=config.get('freeze_backbone', False),
        use_behavior_expert=config.get('use_behavior_expert', False),
    )

    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location=device, weights_only=True)
        incompatible = model.load_state_dict(checkpoint, strict=False)
        logger.info(
            "Initialized from %s (missing=%s, unexpected=%s)",
            args.init_checkpoint,
            list(incompatible.missing_keys),
            list(incompatible.unexpected_keys),
        )

    if config.get('freeze_backbone', False):
        trainable_prefixes = (
            'metadata_expert', 'raw_metadata_expert', 'four_expert_heads',
            'behavior_expert', 'four_router', 'four_residual', 'pairwise_head', 'four_residual_scale'
        )
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith(trainable_prefixes)
        logger.info('Frozen BotDMM trunk; training only four-expert residual modules.')

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    trainer = Trainer(model, config, device)
    best_val_acc, best_model_path = trainer.train(train_loader, val_loader, config['epochs'])

    logger.info("\n" + "=" * 60)
    logger.info("Testing on holdout test set...")
    logger.info("=" * 60)

    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    model.to(device)
    test_metrics = trainer.evaluate(test_loader)

    f1_type = "binary" if num_classes == 2 else "macro"
    logger.info(f"Test Loss: {test_metrics['loss']:.4f}")
    logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"Test Precision ({f1_type}): {test_metrics['precision']:.4f}")
    logger.info(f"Test Recall ({f1_type}): {test_metrics['recall']:.4f}")
    logger.info(f"Test F1 ({f1_type}): {test_metrics['f1']:.4f}")
    logger.info(f"Test MCC: {test_metrics['mcc']:.4f}")

    return test_metrics


def main():
    parser = argparse.ArgumentParser(description="Train the CBAF-Net social intelligent-account detector")

    parser.add_argument("--data_dir", type=str, default="Dataset/Twibot20",
                        help="Data directory; pass the formal Twibot22/Quadbot view explicitly")
    parser.add_argument("--save_dir", type=str, default="./models_saved", help="Model save directory")
    parser.add_argument("--dataset", type=str, default="twibot22", choices=["twibot22", "twibot20", "botsim", "quadbot"],
                        help="Dataset: twibot22 (3-class); twibot20 is a legacy alias; botsim is binary; quadbot is 4-class")
    parser.add_argument("--model_variant", type=str, default="cbaf",
                        choices=[
                            "aemp", "base", "no_memory", "no_event", "no_prototype", "no_domain",
                            "botdmm", "cbaf", "cbaf_net", "quadfusion",
                            "raep_chdf", "raep_only", "chdf_only", "hcrp",
                            "four_expert", "four_expert_behavior", "four_expert_tabular", "four_expert_raep",
                            "four_expert_behavior",
                            "no_structure", "no_content", "no_orthogonal",
                        ],
                        help="Formal model is cbaf; quadfusion/aemp are legacy aliases")

    parser.add_argument("--alpha", type=float, default=0.5, help="Orthogonal constraint coefficient")
    parser.add_argument("--embedding_dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--feature_dim", type=int, default=128, help="Feature dimension")
    parser.add_argument("--num_steps", type=int, default=5, help="Number of temporal steps")
    parser.add_argument("--dropout", type=float, default=None, help="Dropout rate")

    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=None, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience")
    parser.add_argument("--clip_grad", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--val_metric", type=str, default="accuracy", choices=["accuracy", "f1", "mcc"],
                        help="Validation metric used to save the best checkpoint")
    parser.add_argument("--label_smoothing", type=float, default=None,
                        help="Label smoothing for cross-entropy")
    parser.add_argument("--class_weight", type=str, default="none", choices=["none", "balanced", "traditional_up"],
                        help="Class weighting strategy for cross-entropy")
    parser.add_argument("--focal_gamma", type=float, default=0.0,
                        help="Focal loss gamma; 0 disables focal loss")
    parser.add_argument("--init_checkpoint", type=str, default=None,
                        help="Optional checkpoint used to initialize the model")
    parser.add_argument("--include_initial_checkpoint", action="store_true",
                        help="Evaluate and keep the initialized model as a candidate best checkpoint")

    parser.add_argument("--lambda_feature_contrast", type=float, default=0.02, help="Feature contrastive loss weight")
    parser.add_argument("--lambda_class_contrast", type=float, default=0.02, help="Class contrastive loss weight")
    parser.add_argument("--lambda_prototype", type=float, default=0.02, help="Prototype alignment loss weight")
    parser.add_argument("--lambda_domain", type=float, default=0.0, help="Source-domain adversarial loss weight")
    parser.add_argument("--lambda_hierarchy", type=float, default=0.0, help="Hierarchical auxiliary loss weight")
    parser.add_argument("--lambda_consistency", type=float, default=0.0,
                        help="Consistency loss weight between flat and hierarchical predictions")
    parser.add_argument("--lambda_moe_balance", type=float, default=0.01,
                        help="Load-balancing loss weight for QuadMoE-DMM gates")
    parser.add_argument("--temperature", type=float, default=0.1, help="Contrastive learning temperature")
    parser.add_argument("--grl_lambda", type=float, default=1.0, help="Gradient reversal strength")
    parser.add_argument("--hcrp_route_mix", type=float, default=0.45,
                        help="Weight of reliability-routed logits in HCRP")
    parser.add_argument("--hcrp_hierarchy_mix", type=float, default=0.35,
                        help="Weight of hierarchy-composed probabilities in HCRP")
    parser.add_argument("--four_expert_mix", type=float, default=0.15,
                        help="Maximum residual-logit scale for the four-expert extension")
    parser.add_argument("--four_warmup_scale", type=float, default=0.25,
                        help="Initial tanh-scale applied after preserving the BotDMM checkpoint")
    parser.add_argument("--lambda_four_expert", type=float, default=0.10,
                        help="Auxiliary loss weight for four-expert heads")
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Freeze the pretrained BotDMM trunk while fitting the residual experts")
    parser.add_argument("--raw_tabular", action="store_true",
                        help="Preserve raw count/style scales for the tabular expert instead of per-row normalization")
    parser.add_argument("--out", type=str, default=None,
                        help="Optional JSON output path for formal CBAF-Net runs")
    parser.add_argument("--save_embeddings", type=str, default=None,
                        help="Optional embedding output directory for formal CBAF-Net runs")
    parser.add_argument("--cbaf_hidden_dim", "--quadfusion_hidden_dim", dest="cbaf_hidden_dim", type=int, default=192,
                        help="Hidden dimension for CBAF-Net; legacy --quadfusion_hidden_dim is accepted")
    parser.add_argument("--gate_entropy_weight", type=float, default=0.03,
                        help="Gate entropy regularization weight for CBAF-Net")
    parser.add_argument("--contrastive_weight", type=float, default=0.03,
                        help="Supervised contrastive weight for CBAF-Net")
    parser.add_argument("--contrastive_temperature", type=float, default=0.08,
                        help="Supervised contrastive temperature for CBAF-Net")
    parser.add_argument("--cbaf_expert_dropout", "--quadfusion_expert_dropout", dest="cbaf_expert_dropout", type=float, default=0.05,
                        help="Expert-level dropout for CBAF-Net; legacy --quadfusion_expert_dropout is accepted")
    parser.add_argument("--cbaf_aux_weight", "--quadfusion_aux_weight", dest="cbaf_aux_weight", type=float, default=0.3,
                        help="Auxiliary classifier weight for CBAF-Net; legacy --quadfusion_aux_weight is accepted")
    parser.add_argument("--cbaf_ablation", "--quadfusion_ablation", dest="cbaf_ablation", type=str, default="full",
                        choices=["full", "no_graph", "no_temporal", "no_style", "no_semantic", "uniform_gate"],
                        help="Ablation mode for CBAF-Net; legacy --quadfusion_ablation is accepted")

    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")

    args = parser.parse_args()
    args.dataset = canonical_dataset_name(args.dataset)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    logger.info(f"Using device: {device}")

    if uses_paper_cbaf(args):
        run_paper_cbaf(args)
        logger.info("\nTraining completed!")
        return

    args.dropout = default_if_none(args.dropout, 0.3)
    args.lr = default_if_none(args.lr, 0.00005)
    args.weight_decay = default_if_none(args.weight_decay, 5e-4)
    args.epochs = default_if_none(args.epochs, 100)
    args.patience = default_if_none(args.patience, 20)
    args.batch_size = default_if_none(args.batch_size, 32)
    args.label_smoothing = default_if_none(args.label_smoothing, 0.0)

    if args.dataset == 'botsim':
        num_classes = 2
    elif args.dataset == 'quadbot':
        num_classes = 4
    else:
        num_classes = 3
    config = {
        'embedding_dim': args.embedding_dim,
        'feature_dim': args.feature_dim,
        'num_steps': args.num_steps,
        'dropout': args.dropout,
        'temperature': args.temperature,
        'alpha': args.alpha,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'epochs': args.epochs,
        'patience': args.patience,
        'clip_grad': args.clip_grad,
        'val_metric': args.val_metric,
        'label_smoothing': args.label_smoothing,
        'focal_gamma': args.focal_gamma,
        'lambda_feature_contrast': args.lambda_feature_contrast,
        'lambda_class_contrast': args.lambda_class_contrast,
        'lambda_prototype': args.lambda_prototype,
        'lambda_domain': args.lambda_domain,
        'lambda_hierarchy': args.lambda_hierarchy,
        'lambda_consistency': args.lambda_consistency,
        'lambda_moe_balance': args.lambda_moe_balance,
        'lambda_four_expert': args.lambda_four_expert,
        'grl_lambda': args.grl_lambda,
        'batch_size': args.batch_size,
        'save_dir': args.save_dir,
        'include_initial_checkpoint': args.include_initial_checkpoint,
        'num_classes': num_classes,
        'dataset': args.dataset,
        'model_variant': args.model_variant,
        'ablation_mode': MODEL_VARIANT_TO_MODE.get(args.model_variant, args.model_variant),
        'enhanced': args.model_variant in {'raep_chdf', 'raep_only', 'chdf_only', 'hcrp', 'four_expert', 'four_expert_raep', 'four_expert_behavior', 'four_expert_tabular'},
        'use_raep': args.model_variant in {'raep_chdf', 'raep_only', 'hcrp', 'four_expert_raep'},
        'use_hierarchy': args.model_variant in {'raep_chdf', 'chdf_only', 'hcrp', 'four_expert_raep'},
        'use_hcrp': args.model_variant == 'hcrp',
        'use_four_expert': args.model_variant in {'four_expert', 'four_expert_raep', 'four_expert_behavior', 'four_expert_tabular'},
        'use_behavior_expert': args.model_variant == 'four_expert_behavior',
        'raw_tabular': args.raw_tabular or args.model_variant == 'four_expert_tabular',
        'four_expert_mix': args.four_expert_mix,
        'four_warmup_scale': args.four_warmup_scale,
        'freeze_backbone': args.freeze_backbone,
        'hcrp_route_mix': args.hcrp_route_mix,
        'hcrp_hierarchy_mix': args.hcrp_hierarchy_mix,
    }

    os.makedirs(args.save_dir, exist_ok=True)
    train(args, config, device)
    logger.info("\nTraining completed!")


if __name__ == "__main__":
    main()

