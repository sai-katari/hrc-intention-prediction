"""
Synthetic data generator for pipeline testing.

Produces trials with the same structure as real MoGaze data so the full
pipeline (model, training, evaluation) can be verified without the real
dataset. Gaze shifts toward the target object before the wrist trajectory
commits, matching the causal structure reported in the MoGaze literature.

The core generator is pure numpy. TorchMoGazeDataset is a thin wrapper
for use with DataLoader during training.
"""

import numpy as np

N_JOINTS            = 21
JOINT_DIM           = 3
GAZE_DIM            = 3
N_CANDIDATE_OBJECTS = 6
WRIST_JOINT_IDX     = 7


class SyntheticMoGazeGenerator:
    def __init__(self, n_trials=1200, seq_len=90, seed=0):
        self.n_trials = n_trials
        self.seq_len  = seq_len
        self.seed     = seed

    def __len__(self):
        return self.n_trials

    def _goal_layout(self, rng):
        angles = np.linspace(-0.6, 0.6, N_CANDIDATE_OBJECTS)
        goals  = np.stack(
            [0.8 * np.sin(angles), np.full_like(angles, 1.1), 0.8 * np.cos(angles)],
            axis=-1,
        ).astype(np.float32)
        goals += rng.normal(0, 0.02, goals.shape).astype(np.float32)
        return goals

    def __getitem__(self, idx):
        rng    = np.random.default_rng(self.seed * 1_000_003 + idx)
        T      = self.seq_len
        goals  = self._goal_layout(rng)
        label  = int(rng.integers(0, N_CANDIDATE_OBJECTS))
        target = goals[label]

        # gaze: shifts toward target at ~22-40% of the trial
        gaze        = np.zeros((T, GAZE_DIM), dtype=np.float32)
        neutral_dir = np.array([0.0, -0.1, 1.0])
        neutral_dir /= np.linalg.norm(neutral_dir)
        target_dir  = target / np.linalg.norm(target)
        gaze_shift  = int(T * rng.uniform(0.22, 0.40))
        ramp        = max(1, int(T * 0.08))
        for t in range(T):
            alpha  = 0.0 if t < gaze_shift else min(1.0, (t - gaze_shift) / ramp)
            g      = (1 - alpha) * neutral_dir + alpha * target_dir
            g     += rng.normal(0, 0.02, GAZE_DIM)
            gaze[t] = g / np.linalg.norm(g)

        # motion: wrist drifts toward target at ~45-65% of the trial
        motion       = rng.normal(0, 0.01, (T, N_JOINTS, JOINT_DIM)).astype(np.float32)
        start_pos    = np.array([0.3, 0.5, 0.3])
        action_start = int(T * rng.uniform(0.45, 0.65))
        move_ramp    = max(1, int(T * 0.30))
        for t in range(T):
            beta = 0.0 if t < action_start else min(1.0, (t - action_start) / move_ramp)
            motion[t, WRIST_JOINT_IDX] += (1 - beta) * start_pos + beta * target

        motion    = motion.reshape(T, N_JOINTS * JOINT_DIM)
        wrist_xyz = motion.reshape(T, N_JOINTS, JOINT_DIM)[:, WRIST_JOINT_IDX, :].copy()

        return {
            "motion":         motion,
            "gaze":           gaze,
            "wrist_xyz":      wrist_xyz,
            "goal_positions": goals,
            "label":          label,
            "t_action_start": action_start,
        }

    def as_arrays(self):
        motions, gazes, goals, labels, starts = [], [], [], [], []
        for i in range(self.n_trials):
            s = self[i]
            motions.append(s["motion"])
            gazes.append(s["gaze"])
            goals.append(s["goal_positions"])
            labels.append(s["label"])
            starts.append(s["t_action_start"])
        return (
            np.stack(motions), np.stack(gazes), np.stack(goals),
            np.array(labels), np.array(starts),
        )


try:
    import torch
    from torch.utils.data import Dataset

    class TorchMoGazeDataset(Dataset):
        def __init__(self, n_trials=1200, seq_len=90, seed=0):
            self.gen = SyntheticMoGazeGenerator(n_trials, seq_len, seed)

        def __len__(self):
            return len(self.gen)

        def __getitem__(self, idx):
            s = self.gen[idx]
            return {
                "motion":         torch.from_numpy(s["motion"]),
                "gaze":           torch.from_numpy(s["gaze"]),
                "wrist_xyz":      torch.from_numpy(s["wrist_xyz"]),
                "goal_positions": torch.from_numpy(s["goal_positions"]),
                "label":          torch.tensor(s["label"], dtype=torch.long),
                "t_action_start": torch.tensor(s["t_action_start"], dtype=torch.long),
            }

except ImportError:
    TorchMoGazeDataset = None


if __name__ == "__main__":
    gen    = SyntheticMoGazeGenerator(n_trials=5, seq_len=90)
    sample = gen[0]
    for k, v in sample.items():
        print(k, np.shape(v) if hasattr(v, "shape") else v)