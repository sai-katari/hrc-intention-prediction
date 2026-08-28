"""
Sanity check for the synthetic data generator.

Trains a RandomForest on flattened observation prefixes and checks that
gaze becomes predictive before motion - the qualitative pattern reported
in MoGaze. If this doesn't hold, the generator's causal structure needs
fixing before moving to the full model.

Run this before training on synthetic data:
    python data/sanity_check_baseline.py
"""

import os
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(__file__))
from synthetic_generator import N_CANDIDATE_OBJECTS, SyntheticMoGazeGenerator


def flatten_prefix(motion, gaze, frac, modality="both"):
    T      = motion.shape[1]
    cutoff = max(2, int(T * frac))
    m = motion[:, :cutoff].reshape(motion.shape[0], -1)
    g = gaze[:, :cutoff].reshape(gaze.shape[0], -1)
    if modality == "motion":
        return m
    if modality == "gaze":
        return g
    return np.concatenate([m, g], axis=1)


def run():
    train_gen = SyntheticMoGazeGenerator(n_trials=1000, seq_len=90, seed=1)
    test_gen  = SyntheticMoGazeGenerator(n_trials=300,  seq_len=90, seed=2)
    motion_tr, gaze_tr, _, y_tr, _ = train_gen.as_arrays()
    motion_te, gaze_te, _, y_te, _ = test_gen.as_arrays()

    fractions = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    print(f"{'frac':>6} | {'gaze':>8} | {'motion':>8} | {'both':>8}")
    print("-" * 40)
    for frac in fractions:
        row = []
        for modality in ["gaze", "motion", "both"]:
            Xtr = flatten_prefix(motion_tr, gaze_tr, frac, modality)
            Xte = flatten_prefix(motion_te, gaze_te, frac, modality)
            clf = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=0)
            clf.fit(Xtr, y_tr)
            row.append(clf.score(Xte, y_te))
        print(f"{frac:>6.1f} | {row[0]:>8.3f} | {row[1]:>8.3f} | {row[2]:>8.3f}")

    print(f"\nChance: {1.0 / N_CANDIDATE_OBJECTS:.3f}")
    print("Expected: gaze beats motion at low fractions, both converge near 1.0 at the end.")


if __name__ == "__main__":
    run()