"""
Baseline comparison for anytime intention prediction on MoGaze.

Evaluates four baselines against the trained CNN-Transformer:
    Random         - uniform over n_goals
    Gaze-direction - predict the object most aligned with gaze
    Wrist-distance - predict the closest object to the wrist
    LSTM           - trained on the same LOPO split as the main model

Usage:
    python baselines.py --mogaze_path data/mogaze --n_goals 10 \
                        --motion_dim 66 --test_pid p1
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


# --- Heuristic baselines ---

def gaze_direction_baseline(batch, frac):
    gaze      = batch["gaze"].numpy()
    wrist_xyz = batch["wrist_xyz"].numpy()
    goals     = batch["goal_positions"].numpy()
    t = max(0, int(gaze.shape[1] * frac) - 1)
    preds = []
    for b in range(len(gaze)):
        to_goal = goals[b] - wrist_xyz[b, t:t+1]
        to_goal_n = to_goal / (np.linalg.norm(to_goal, axis=1, keepdims=True) + 1e-6)
        gaze_n = gaze[b, t] / (np.linalg.norm(gaze[b, t]) + 1e-6)
        preds.append((to_goal_n * gaze_n).sum(axis=1).argmax())
    return np.array(preds)


def wrist_distance_baseline(batch, frac):
    wrist_xyz = batch["wrist_xyz"].numpy()
    goals     = batch["goal_positions"].numpy()
    t = max(0, int(wrist_xyz.shape[1] * frac) - 1)
    preds = []
    for b in range(len(wrist_xyz)):
        dist = np.linalg.norm(goals[b] - wrist_xyz[b, t:t+1], axis=1)
        preds.append(dist.argmin())
    return np.array(preds)


def evaluate_heuristic(loader, baseline_fn, frac):
    correct, total = 0, 0
    for batch in loader:
        labels = batch["label"].numpy()
        preds  = baseline_fn(batch, frac)
        correct += (preds == labels).sum()
        total   += len(labels)
    return correct / total


# --- LSTM baseline ---

class LSTMIntentionModel(nn.Module):
    """
    Unidirectional LSTM with the same input features as the Transformer
    (motion + gaze + goal-relative), but no CNN stems.
    """
    def __init__(self, motion_dim=66, gaze_dim=3, n_goals=10,
                 hidden_size=256, num_layers=2, dropout=0.1):
        super().__init__()
        self.n_goals = n_goals
        input_dim = motion_dim + gaze_dim + n_goals * 2
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.classifier = nn.Linear(hidden_size, n_goals)

    @staticmethod
    def goal_features(wrist_xyz, gaze, goal_positions):
        gaze_n    = gaze / (gaze.norm(dim=-1, keepdim=True) + 1e-6)
        goals_exp = goal_positions.unsqueeze(1)
        to_goal   = goals_exp - wrist_xyz.unsqueeze(2)
        to_goal_n = to_goal / (to_goal.norm(dim=-1, keepdim=True) + 1e-6)
        align     = (gaze_n.unsqueeze(2) * to_goal_n).sum(-1)
        proximity = -to_goal.norm(dim=-1)
        return torch.cat([align, proximity], dim=-1)

    def forward(self, motion, gaze, goal_positions, wrist_xyz):
        gr = self.goal_features(wrist_xyz, gaze, goal_positions)
        h, _ = self.lstm(torch.cat([motion, gaze, gr], dim=-1))
        return self.classifier(h)


def train_lstm(train_ds, motion_dim, n_goals, epochs=20, device="cpu",
               val_frac=0.15, seed=42):
    """
    Train the LSTM using only training-participant data.
    An inner validation split (15% of training trials) is used for
    checkpoint selection — the held-out test participant is never used here.
    """
    n = len(train_ds)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    cut = int(n * (1 - val_frac))
    inner_train = Subset(train_ds, idx[:cut].tolist())
    inner_val   = Subset(train_ds, idx[cut:].tolist())

    model = LSTMIntentionModel(motion_dim=motion_dim, n_goals=n_goals).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    best, best_state = -1.0, None

    for epoch in range(epochs):
        model.train()
        for batch in DataLoader(inner_train, batch_size=32, shuffle=True):
            motion    = batch["motion"].to(device)
            gaze      = batch["gaze"].to(device)
            wrist_xyz = batch["wrist_xyz"].to(device)
            goals     = batch["goal_positions"].to(device)
            labels    = batch["label"].to(device)
            logits    = model(motion, gaze, goals, wrist_xyz)
            B, T, C   = logits.shape
            loss = nn.functional.cross_entropy(
                logits.reshape(B * T, C),
                labels.unsqueeze(1).expand(B, T).reshape(B * T),
            )
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in DataLoader(inner_val, batch_size=32):
                motion    = batch["motion"].to(device)
                gaze      = batch["gaze"].to(device)
                wrist_xyz = batch["wrist_xyz"].to(device)
                goals     = batch["goal_positions"].to(device)
                labels    = batch["label"].to(device)
                logits    = model(motion, gaze, goals, wrist_xyz)
                t50 = max(0, int(logits.size(1) * 0.5) - 1)
                correct += (logits[:, t50].argmax(-1) == labels).sum().item()
                total   += labels.size(0)
        acc = correct / total
        if acc > best:
            best, best_state = acc, {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"  LSTM epoch {epoch+1:2d} | inner_val@0.5: {acc:.3f}", flush=True)

    model.load_state_dict(best_state)
    return model


def evaluate_lstm(model, loader, device):
    model.eval()
    correct = {f: 0 for f in FRACTIONS}
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
            for f in FRACTIONS:
                idx = max(0, int(T * f) - 1)
                correct[f] += (logits[:, idx].argmax(-1) == labels).sum().item()
    return {f: correct[f] / total for f in FRACTIONS}


# --- Main ---

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mogaze_path", required=True)
    ap.add_argument("--n_goals",     type=int, default=10)
    ap.add_argument("--motion_dim",  type=int, default=66)
    ap.add_argument("--test_pid",    type=str, default=None)
    ap.add_argument("--lstm_epochs", type=int, default=20)
    args = ap.parse_args()

    from data.mogaze_loader import MoGazeDataset
    train_ds    = MoGazeDataset(args.mogaze_path, split="train", lopo_test_pid=args.test_pid)
    test_ds     = MoGazeDataset(args.mogaze_path, split="test",  lopo_test_pid=args.test_pid)
    test_loader = DataLoader(test_ds, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chance = 1.0 / args.n_goals

    print("Evaluating heuristic baselines...")
    gaze_accs  = {f: evaluate_heuristic(test_loader, gaze_direction_baseline, f) for f in FRACTIONS}
    wrist_accs = {f: evaluate_heuristic(test_loader, wrist_distance_baseline, f) for f in FRACTIONS}

    print("Training LSTM baseline...")
    lstm_model = train_lstm(train_ds, args.motion_dim, args.n_goals,
                            epochs=args.lstm_epochs, device=str(device))
    lstm_accs = evaluate_lstm(lstm_model, test_loader, device)

    print(f"\n{'Frac':>5} | {'Random':>8} | {'Gaze-Dir':>10} | {'Wrist-Dist':>12} | {'LSTM':>8}")
    print("-" * 55)
    for f in FRACTIONS:
        print(f"{f:>5.1f} | {chance:>8.3f} | {gaze_accs[f]:>10.3f} | "
              f"{wrist_accs[f]:>12.3f} | {lstm_accs[f]:>8.3f}")

    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(FRACTIONS, [chance] * len(FRACTIONS),        ":",  color="gray", label="Random")
    plt.plot(FRACTIONS, [gaze_accs[f]  for f in FRACTIONS], "--s", label="Gaze-direction")
    plt.plot(FRACTIONS, [wrist_accs[f] for f in FRACTIONS], "--^", label="Wrist-distance")
    plt.plot(FRACTIONS, [lstm_accs[f]  for f in FRACTIONS], "-D",  label="LSTM")
    plt.xlabel("Fraction of trial observed")
    plt.ylabel("Top-1 accuracy")
    plt.title("Baseline Comparison")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig("results/baseline_comparison.png", dpi=150)
    print("Saved results/baseline_comparison.png")


if __name__ == "__main__":
    main()