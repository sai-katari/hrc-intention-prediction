"""
Evaluate a trained checkpoint across four modality conditions.
Reports top-1 accuracy, top-3 accuracy, and macro-F1 at each observation fraction.

Ablation conditions:
    Full             - Motion + Gaze + Goal-relative features
    No goal features - Motion + Gaze only (goal branch zeroed via model flag)
    Motion only      - Gaze zeroed everywhere
    Gaze only        - Motion zeroed + wrist_xyz zeroed (wrist comes from FK)

Usage:
    python evaluate.py --n_goals 10 --motion_dim 66 \
        --mogaze_path data/mogaze --test_pid p1
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from data.synthetic_generator import TorchMoGazeDataset
from models.cnn_transformer import IntentionTransformer

FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def eval_modality(model, loader, device,
                  zero_motion=False, zero_gaze=False, use_goal_features=True):
    model.eval()
    correct_top1 = {f: 0 for f in FRACTIONS}
    correct_top3 = {f: 0 for f in FRACTIONS}
    all_preds    = {f: [] for f in FRACTIONS}
    all_labels   = []
    total = 0

    with torch.no_grad():
        for batch in loader:
            motion    = batch["motion"].to(device)
            gaze      = batch["gaze"].to(device)
            wrist_xyz = batch["wrist_xyz"].to(device)
            goals     = batch["goal_positions"].to(device)
            labels    = batch["label"].to(device)

            if zero_motion:
                motion    = torch.zeros_like(motion)
                wrist_xyz = torch.zeros_like(wrist_xyz)  # wrist is FK-derived from motion

            if zero_gaze:
                gaze = torch.zeros_like(gaze)

            logits = model(motion, gaze, goals, wrist_xyz, use_goal_features=use_goal_features)
            T = logits.size(1)
            total += labels.size(0)
            all_labels.extend(labels.cpu().tolist())

            for f in FRACTIONS:
                step = logits[:, max(0, int(T * f) - 1), :]
                top1 = step.argmax(dim=-1)
                top3 = step.topk(min(3, step.size(-1)), dim=-1).indices
                correct_top1[f] += (top1 == labels).sum().item()
                correct_top3[f] += sum(labels[i].item() in top3[i].tolist() for i in range(labels.size(0)))
                all_preds[f].extend(top1.cpu().tolist())

    n_goals    = logits.size(-1)
    labels_arr = np.array(all_labels)
    results    = {}
    for f in FRACTIONS:
        preds = np.array(all_preds[f])
        f1s   = []
        for c in range(n_goals):
            tp   = ((preds == c) & (labels_arr == c)).sum()
            fp   = ((preds == c) & (labels_arr != c)).sum()
            fn   = ((preds != c) & (labels_arr == c)).sum()
            prec = tp / (tp + fp + 1e-8)
            rec  = tp / (tp + fn + 1e-8)
            f1s.append(2 * prec * rec / (prec + rec + 1e-8))
        results[f] = {
            "top1": correct_top1[f] / total,
            "top3": correct_top3[f] / total,
            "macro_f1": float(np.mean(f1s)),
        }
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",  default="checkpoints/intention_transformer.pt")
    ap.add_argument("--out",         default="results/accuracy_vs_time.png")
    ap.add_argument("--n_goals",     type=int, default=6)
    ap.add_argument("--motion_dim",  type=int, default=63)
    ap.add_argument("--mogaze_path", type=str, default=None)
    ap.add_argument("--test_pid",    type=str, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = IntentionTransformer(motion_dim=args.motion_dim, n_goals=args.n_goals).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    if args.mogaze_path:
        from data.mogaze_loader import MoGazeDataset
        test_ds = MoGazeDataset(args.mogaze_path, split="test", lopo_test_pid=args.test_pid)
    else:
        test_ds = TorchMoGazeDataset(n_trials=500, seq_len=90, seed=99)

    loader = DataLoader(test_ds, batch_size=32)

    conditions = [
        ("Full (Motion+Gaze+Goal)", dict()),
        ("Motion+Gaze (no goal)",   dict(use_goal_features=False)),
        ("Gaze only",               dict(zero_motion=True)),
        ("Motion only",             dict(zero_gaze=True)),
    ]

    all_results = {name: eval_modality(model, loader, device, **kw) for name, kw in conditions}

    # Plot
    chance = 1.0 / args.n_goals
    plt.figure(figsize=(8, 5))
    for (name, _), style in zip(conditions, ["-o", "-D", "--s", "--^"]):
        plt.plot(FRACTIONS, [all_results[name][f]["top1"] for f in FRACTIONS], style, label=name)
    plt.axhline(chance, color="gray", linestyle=":", label=f"Chance ({args.n_goals} objects)")
    plt.xlabel("Fraction of trial observed")
    plt.ylabel("Top-1 accuracy")
    title = "Modality Ablation"
    if args.test_pid:
        title += f" — LOPO test={args.test_pid}"
    plt.title(title)
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")

    # Table
    print(f"\n{'Frac':>5} | {'Full Top1':>10} {'Top3':>6} {'F1':>6} | "
          f"{'No Goal':>8} | {'Gaze':>6} | {'Motion':>8}")
    print("-" * 65)
    for f in FRACTIONS:
        r = all_results
        print(f"{f:>5.1f} | "
              f"{r['Full (Motion+Gaze+Goal)'][f]['top1']:>10.3f} "
              f"{r['Full (Motion+Gaze+Goal)'][f]['top3']:>6.3f} "
              f"{r['Full (Motion+Gaze+Goal)'][f]['macro_f1']:>6.3f} | "
              f"{r['Motion+Gaze (no goal)'][f]['top1']:>8.3f} | "
              f"{r['Gaze only'][f]['top1']:>6.3f} | "
              f"{r['Motion only'][f]['top1']:>8.3f}")
    print(f"\nChance: {chance:.3f}")


if __name__ == "__main__":
    main()