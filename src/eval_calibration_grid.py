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
    args = p.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = [
        load_model(args.dataset, args.data_dir, args.base, 'base', device),
        load_model(args.dataset, args.data_dir, args.raep, 'raep', device),
        load_model(args.dataset, args.data_dir, args.four, 'four', device),
    ]
    val = [predict(m, args.dataset, args.data_dir, 'val', device) for m in models]
    test = [predict(m, args.dataset, args.data_dir, 'test', device) for m in models]
    yv, yt = val[0][1].numpy(), test[0][1].numpy()
    vp = [torch.softmax(x[0], dim=1) for x in val]
    tp = [torch.softmax(x[0], dim=1) for x in test]
    k = vp[0].shape[1]
    best = None
    for a in range(21):
        for b in range(21 - a):
            c = 20 - a - b
            w = [a / 20.0, b / 20.0, c / 20.0]
            pv = sum(wi * pi for wi, pi in zip(w, vp)).clamp_min(1e-8).log()
            pt = sum(wi * pi for wi, pi in zip(w, tp)).clamp_min(1e-8).log()
            for bh in np.arange(-0.4, 0.401, 0.05):
                for bt in np.arange(-0.4, 0.401, 0.05):
                    bias = torch.zeros(k)
                    bias[0] = float(bh)
                    bias[1] = float(bt)
                    score = f1_score(yv, (pv + bias).argmax(1).numpy(), average='macro')
                    if best is None or score > best['val_f1']:
                        pred_t = (pt + bias).argmax(1).numpy()
                        best = {
                            'weights': w,
                            'bias_human': float(bh),
                            'bias_traditional': float(bt),
                            'val_f1': float(score),
                            'test_f1': float(f1_score(yt, pred_t, average='macro')),
                            'dataset': args.dataset,
                        }
    Path(args.out).write_text(json.dumps(best, indent=2))
    print(json.dumps(best, indent=2))


if __name__ == '__main__':
    main()
