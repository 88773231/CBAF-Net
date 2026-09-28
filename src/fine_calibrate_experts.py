import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval_ensemble_grid import load_model, predict


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, choices=['quadbot', 'twibot20'])
    p.add_argument('--data_dir', required=True)
    p.add_argument('--base', required=True)
    p.add_argument('--raep', required=True)
    p.add_argument('--four', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--trials', type=int, default=20000)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = [
        load_model(args.dataset, args.data_dir, args.base, 'base', device),
        load_model(args.dataset, args.data_dir, args.raep, 'raep', device),
        load_model(args.dataset, args.data_dir, args.four, 'four', device),
    ]
    val = [predict(m, args.dataset, args.data_dir, 'val', device) for m in models]
    test = [predict(m, args.dataset, args.data_dir, 'test', device) for m in models]
    yv = val[0][1].numpy()
    yt = test[0][1].numpy()
    vp = torch.stack([torch.softmax(x[0], dim=1) for x in val]).numpy()
    tp = torch.stack([torch.softmax(x[0], dim=1) for x in test]).numpy()
    classes = vp.shape[-1]

    rng = np.random.default_rng(20260924)
    candidates = []
    # Include the previously observed calibration point and the uncalibrated
    # baseline as deterministic candidates.
    candidates.append((np.array([0.0, 0.7, 0.3]), np.array([0.15, 0.05] + [0.0] * (classes - 2)), np.ones(classes)))
    candidates.append((np.array([1.0, 0.0, 0.0]), np.zeros(classes), np.ones(classes)))
    for _ in range(args.trials):
        w = rng.dirichlet(np.ones(3) * 1.2)
        bias = np.clip(rng.normal(0.0, 0.14, size=classes), -0.45, 0.45)
        scale = np.clip(rng.lognormal(0.0, 0.14, size=classes), 0.70, 1.35)
        candidates.append((w, bias, scale))

    best = None
    for w, bias, scale in candidates:
        val_prob = np.clip(np.sum(w[:, None, None] * vp, axis=0), 1e-8, 1.0)
        val_logits = np.log(val_prob) * scale[None, :] + bias[None, :]
        score = f1_score(yv, val_logits.argmax(axis=1), average='macro')
        if best is None or score > best['val_f1']:
            test_prob = np.clip(np.sum(w[:, None, None] * tp, axis=0), 1e-8, 1.0)
            test_logits = np.log(test_prob) * scale[None, :] + bias[None, :]
            best = {
                'weights': [float(x) for x in w],
                'bias': [float(x) for x in bias],
                'scale': [float(x) for x in scale],
                'val_f1': float(score),
                'test_f1': float(f1_score(yt, test_logits.argmax(axis=1), average='macro')),
                'dataset': args.dataset,
                'trials': len(candidates),
            }
    Path(args.out).write_text(json.dumps(best, indent=2))
    print(json.dumps(best, indent=2))


if __name__ == '__main__':
    main()
