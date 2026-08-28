"""
Full LOPO cross-validation across all participants with gaze data.

Trains a separate model for each held-out participant and reports
per-fold accuracy plus mean ± std across folds.

Usage:
    python run_lopo.py --mogaze_path data/mogaze --n_goals 10 --motion_dim 66

"""

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

ALL_PIDS  = ["p1", "p2", "p4", "p5", "p6", "p7"]  # p3 has no gaze file
FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def train_one_fold(train_ds, val_ds, motion_dim, n_goals, epochs, device):
    from models.cnn_transformer import IntentionTransformer
    from train import evaluate, prefix_weighted_loss

    model = IntentionTransformer(motion_dim=motion_dim, n_goals=n_goals).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=32)

    informative = [0.2, 0.3, 0.4, 0.5]
    best_score, best_state = -1.0, None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            motion    = batch["motion"].to(device)
            gaze      = batch["gaze"].to(device)
            wrist_xyz = batch["wrist_xyz"].to(device)
            goals     = batch["goal_positions"].to(device)
            labels    = batch["label"].to(device)
            logits    = model(motion, gaze, goals, wrist_xyz)
            loss      = prefix_weighted_loss(logits, labels, epoch, epochs)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
        sched.step()

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            accs  = evaluate(model, val_loader, device)
            score = sum(accs[f] for f in informative) / len(informative)
            if score > best_score:
                best_score = score
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"    epoch {epoch+1:3d} | loss {total_loss/len(train_loader):.4f} "
                  f"| score {score:.3f}", flush=True)

    model.load_state_dict(best_state)
    return model


def evaluate_model(model, loader, device):
    model.eval()
    correct = {f: 0 for f in FRACTIONS}
    total   = 0
    with torch.no_grad():
        for batch in loader:
            motion    = batch["motion"].to(device)
            gaze      = batch["gaze"].to(device)
            wrist_xyz = batch["wrist_xyz"].to(device)
            goals     = batch["goal_positions"].to(device)
            labels    = batch["label"].to(device)
            logits    = model(motion, gaze, goals, wrist_xyz)
            T = logits.size(1)
            total += labels.size(0)
            for f in FRACTIONS:
                idx = max(0, int(T * f) - 1)
                correct[f] += (logits[:, idx].argmax(-1) == labels).sum().item()
    return {f: correct[f] / total for f in FRACTIONS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mogaze_path", required=True)
    ap.add_argument("--n_goals",    type=int,   default=10)
    ap.add_argument("--motion_dim", type=int,   default=66)
    ap.add_argument("--epochs",     type=int,   default=30)
    ap.add_argument("--pids",       nargs="+",  default=ALL_PIDS)
    ap.add_argument("--out",        default="results/lopo_results.json")
    args = ap.parse_args()

    from data.mogaze_loader import MoGazeDataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("results", exist_ok=True)

    fold_results = {}
    for test_pid in args.pids:
        print(f"\n{'='*55}\nLOPO fold: test_pid = {test_pid}\n{'='*55}")

        train_ds = MoGazeDataset(args.mogaze_path, split="train", lopo_test_pid=test_pid)
        test_ds  = MoGazeDataset(args.mogaze_path, split="test",  lopo_test_pid=test_pid)

        if len(test_ds) == 0:
            print(f"  No test trials for {test_pid}, skipping.")
            continue

        # Inner val split — test participant never used for model selection
        val_size   = max(1, int(0.15 * len(train_ds)))
        train_size = len(train_ds) - val_size
        inner_train, inner_val = torch.utils.data.random_split(
            train_ds, [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

        model = train_one_fold(inner_train, inner_val, args.motion_dim,
                               args.n_goals, args.epochs, device)
        accs  = evaluate_model(model, DataLoader(test_ds, batch_size=32), device)

        fold_results[test_pid] = {str(f): accs[f] for f in FRACTIONS}
        print("  " + " ".join(f"{f:.1f}:{accs[f]:.3f}" for f in FRACTIONS))

        with open(args.out, "w") as fh:
            json.dump(fold_results, fh, indent=2)

    if not fold_results:
        print("No folds completed.")
        return

    print(f"\n{'='*55}\nLOPO Summary\n{'='*55}")
    print(f"{'Frac':>5} | {'Mean':>8} | {'Std':>7} | {'Min':>7} | {'Max':>7}")
    print("-" * 45)
    for f in FRACTIONS:
        vals = [fold_results[pid][str(f)] for pid in fold_results]
        print(f"{f:>5.1f} | {np.mean(vals):>8.3f} | {np.std(vals):>7.3f} | "
              f"{min(vals):>7.3f} | {max(vals):>7.3f}")
    print(f"\nFolds: {list(fold_results.keys())}")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()