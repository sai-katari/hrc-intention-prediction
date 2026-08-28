"""
Print the structure of one MoGaze session's HDF5 files.

Useful for debugging the loader or verifying dataset field names.

Usage (use forward slashes on Windows):
    python data/inspect_mogaze.py "C:/path/to/mogaze"
"""

import csv
import os
import sys


def inspect_hdf5(path):
    import h5py
    print(f"\n=== {os.path.basename(path)} ===")
    with h5py.File(path, "r") as f:
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  dataset: {name}  shape={obj.shape}  dtype={obj.dtype}")
                for k, v in obj.attrs.items():
                    shown = v if not hasattr(v, "shape") or getattr(v, "size", 0) < 20 else f"<array len {v.size}>"
                    print(f"    attr[{k}] = {shown}")
            else:
                print(f"  group: {name}")
                for k, v in obj.attrs.items():
                    print(f"    attr[{k}] = {v}")
        f.visititems(visitor)
        print("  root attrs:")
        for k, v in f.attrs.items():
            print(f"    {k} = {v}")


def inspect_instructions(path):
    print(f"\n=== {os.path.basename(path)} ===")
    try:
        with open(path, newlines="") as fh:
            for i, row in enumerate(csv.reader(fh)):
                print(" ", row)
                if i >= 5:
                    break
        return
    except Exception as e:
        print(f"  not CSV ({e}); trying HDF5")
    try:
        inspect_hdf5(path)
    except Exception as e2:
        print(f"  not HDF5 either ({e2}); first 300 bytes:")
        with open(path, "rb") as fh:
            print(fh.read(300))


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    pid, sess = "p1", "1"

    for label, fname in {
        "human":        f"{pid}_{sess}_human_data.hdf5",
        "gaze":         f"{pid}_{sess}_gaze_data.hdf5",
        "object":       f"{pid}_{sess}_object_data.hdf5",
        "segmentations":f"{pid}_{sess}_segmentations.hdf5",
    }.items():
        fpath = os.path.join(root, fname)
        if os.path.exists(fpath):
            inspect_hdf5(fpath)
        else:
            print(f"  [missing] {fpath}")

    for ext in ["", ".csv", ".hdf5", ".yaml", ".yml", ".json", ".txt"]:
        cand = os.path.join(root, f"{pid}_{sess}_instructions{ext}")
        if os.path.exists(cand):
            inspect_instructions(cand)
            break
    else:
        print(f"\n  [missing] {pid}_{sess}_instructions.*")