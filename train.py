"""
Training script. Run on your own machine with torch + (ideally) a GPU.

"""
import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.synthetic_generator import TorchMoGazeDataset
from models.cnn_transformer import IntentionTransformer


def prefix_weighted_loss(logits, labels, epoch, total_epochs):
    """
    Supervise every timestep with the same label, but exclude very early
    timesteps from the loss. The cutoff anneals from 15% → 10% over the
    first 30% of training, so the model first learns from the informative
    late-trial frames before being asked to predict earlier.
    """
    B, T, C = logits.shape
    labels_exp = labels.unsqueeze(1).expand(B, T)
    losses = nn.functional.cross_entropy(
        logits.reshape(B * T, C), labels_exp.reshape(B * T), reduction="none"
    ).view(B, T)

    t_frac = torch.linspace(0, 1, T, device=logits.device)
    anneal = min(1.0, epoch / max(1, total_epochs * 0.3))
    cutoff = 0.10 + (0.15 - 0.10) * (1 - anneal)
    weight = (t_frac >= cutoff).float()
    weight = weight / weight.sum().clamp(min=1)
    return (losses * weight.unsqueeze(0)).sum(dim=1).mean()


def evaluate(model, loader, device):
    model.eval()
    fractions = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    correct = {f: 0 for f in fractions}
    total = 0
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
            for f in fractions:
                idx = max(0, int(T * f) - 1)
                correct[f] += (logits[:, idx].argmax(-1) == labels).sum().item()
    model.train()
    return {f: correct[f] / total for f in fractions}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs",     type=int,   default=30)
    ap.add_argument("--batch_size", type=int,   default=32)
    ap.add_argument("--lr",         type=float, default=3e-4)
    ap.add_argument("--data",       choices=["synthetic", "mogaze"], default="synthetic")
    ap.add_argument("--mogaze_path",type=str,   default=None)
    ap.add_argument("--n_goals",    type=int,   default=None)
    ap.add_argument("--motion_dim", type=int,   default=None)
    ap.add_argument("--test_pid",   type=str,   default=None,
                    help="LOPO held-out participant (e.g. 'p1')")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.data == "synthetic":
        train_ds   = TorchMoGazeDataset(n_trials=2000, seq_len=90, seed=1)
        val_ds     = TorchMoGazeDataset(n_trials=400,  seq_len=90, seed=2)
        n_goals    = args.n_goals    or 6
        motion_dim = args.motion_dim or 63
    else:
        from data.mogaze_loader import MoGazeDataset
        full_train_ds = MoGazeDataset(args.mogaze_path, split="train",
                                      lopo_test_pid=args.test_pid)
        n_goals    = args.n_goals    or 10
        motion_dim = args.motion_dim or full_train_ds[0]["motion"].shape[-1]

        # Inner val split from training participants only.
        # The held-out test participant is never used for model selection.
        val_size   = max(1, int(0.15 * len(full_train_ds)))
        train_size = len(full_train_ds) - val_size
        train_ds, val_ds = torch.utils.data.random_split(
            full_train_ds, [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size)

    model = IntentionTransformer(motion_dim=motion_dim, n_goals=n_goals).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    os.makedirs("checkpoints", exist_ok=True)

    # Checkpoint selection uses avg accuracy over fractions 0.2–0.5.
    # frac=1.0 saturates early and stops discriminating good checkpoints.
    informative = [0.2, 0.3, 0.4, 0.5]
    best_score  = -1.0

    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch in train_loader:
            motion    = batch["motion"].to(device)
            gaze      = batch["gaze"].to(device)
            wrist_xyz = batch["wrist_xyz"].to(device)
            goals     = batch["goal_positions"].to(device)
            labels    = batch["label"].to(device)

            logits = model(motion, gaze, goals, wrist_xyz)
            loss   = prefix_weighted_loss(logits, labels, epoch, args.epochs)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()

        sched.step()
        avg_loss = total_loss / len(train_loader)

        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            accs    = evaluate(model, val_loader, device)
            acc_str = " ".join(f"{f:.1f}:{a:.3f}" for f, a in accs.items())
            score   = sum(accs[f] for f in informative) / len(informative)
            is_best = score > best_score
            if is_best:
                best_score = score
                torch.save(model.state_dict(), "checkpoints/intention_transformer.pt")
            marker = " <- best, saved" if is_best else ""
            print(f"epoch {epoch+1:3d} | loss {avg_loss:.4f} | acc@frac {acc_str} | score {score:.3f}{marker}")
        else:
            print(f"epoch {epoch+1:3d} | loss {avg_loss:.4f}")

    print(f"\nBest score: {best_score:.3f}")
    print("Checkpoint saved to checkpoints/intention_transformer.pt")


if __name__ == "__main__":
    main()