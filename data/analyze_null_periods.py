"""
Analyze the length of approach periods in the MoGaze dataset.

Usage:
    python data/analyze_null_periods.py "path/to/mogaze"
"""

import glob
import os
import sys

import h5py
import numpy as np


NATIVE_HZ = 120
TARGET_HZ = 30
STRIDE = NATIVE_HZ // TARGET_HZ


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    files = glob.glob(os.path.join(root, "*_segmentations.hdf5"))

    gap_lengths_native = []

    for fpath in sorted(files):
        with h5py.File(fpath, "r") as f:
            segments = f["segments"]

            rows = [
                (
                    int(row["start"]),
                    int(row["end"]),
                    row["label"].decode()
                    if isinstance(row["label"], (bytes, bytearray))
                    else str(row["label"]),
                )
                for row in segments
            ]

        # Identify null segments that directly precede an object interaction.
        for i in range(len(rows) - 1):
            null_start, null_end, null_label = rows[i]
            next_start, _, next_label = rows[i + 1]

            gap = next_start - null_end

            if (
                null_label != "null"
                or next_label == "null"
                or gap < 0
                or gap > 1
            ):
                continue

            gap_lengths_native.append(null_end - null_start)

    gap_lengths_native = np.array(gap_lengths_native)
    gap_lengths_downsampled = gap_lengths_native // STRIDE

    print(
        f"Total null-to-action transitions found: "
        f"{len(gap_lengths_native)}"
    )

    print("\nNative-frame (120 Hz) gap length statistics:")
    print(
        f"  min={gap_lengths_native.min()}  "
        f"max={gap_lengths_native.max()}  "
        f"mean={gap_lengths_native.mean():.1f}  "
        f"median={np.median(gap_lengths_native):.1f}"
    )

    print("\nDownsampled (30 Hz) gap length statistics:")
    print(
        f"  min={gap_lengths_downsampled.min()}  "
        f"max={gap_lengths_downsampled.max()}  "
        f"mean={gap_lengths_downsampled.mean():.1f}  "
        f"median={np.median(gap_lengths_downsampled):.1f}"
    )

    print("\nTrials retained for different sequence lengths:")

    for seq_len in [15, 20, 30, 45, 60, 75, 90]:
        n_ok = (gap_lengths_downsampled >= seq_len).sum()
        pct = 100 * n_ok / len(gap_lengths_downsampled)

        print(
            f"  seq_len={seq_len:3d} "
            f"({seq_len / TARGET_HZ:.1f}s): "
            f"{n_ok:4d}/{len(gap_lengths_downsampled)} "
            f"trials retained ({pct:.0f}%)"
        )


if __name__ == "__main__":
    main()