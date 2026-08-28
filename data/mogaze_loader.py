"""
MoGaze dataset loader.

Requires humoro:
    git clone https://github.com/PhilippJKratzer/humoro.git
    cd humoro && pip install -r requirements.txt && pip install . && cd ..

"""

import os
import glob
import xml.etree.ElementTree as ET
import numpy as np

try:
    import h5py
    import torch
    from torch.utils.data import Dataset
    from humoro.trajectory import Trajectory
    from humoro.kin_pybullet import HumanKin
    from humoro.gaze import load_gaze
    from humoro.objectParser import parseHDF5
    DEPS_OK = True
    _import_error = None
except ImportError as e:
    DEPS_OK = False
    _import_error = e
    Dataset = object

WRIST_JOINT_NAME = "rWristRotZ"
NATIVE_HZ = 120
TARGET_HZ = 30
DOWNSAMPLE_STRIDE = NATIVE_HZ // TARGET_HZ

# 60 frames = ~2s at 30Hz. Keeps 84% of real approach gaps;
# 90 frames (3s) only keeps 47% because many gaps are short.
SEQ_LEN = 60


def find_sessions(root):
    files = glob.glob(os.path.join(root, "*_human_data.hdf5"))
    sessions = []
    for f in files:
        base = os.path.basename(f).replace("_human_data.hdf5", "")
        pid, sess = base.rsplit("_", 1)
        sessions.append((pid, sess))
    return sorted(sessions)


def find_scene_xml(root):
    candidate = os.path.join(root, "scene.xml")
    if os.path.exists(candidate):
        return candidate
    matches = glob.glob(os.path.join(root, "**", "*.xml"), recursive=True)
    if matches:
        print(f"  scene.xml not at top level; using {matches[0]}")
        return matches[0]
    raise FileNotFoundError(f"No scene .xml found under {root}")


def parse_object_names(obj_ids, scene_xml_path):
    """Map numeric body IDs to semantic names using scene.xml."""
    root = ET.parse(scene_xml_path).getroot()
    obj_names = [""] * len(obj_ids)
    for child in root:
        if child.tag != "body":
            continue
        id_ = int(child.attrib["id"])
        if id_ not in obj_ids:
            continue
        obj_names[obj_ids.index(id_)] = child.attrib["name"]
    return obj_names


class MoGazeSession:
    """Loads one participant/session and builds trials."""

    def __init__(self, root, pid, sess, scene_xml=None):
        if not DEPS_OK:
            raise ImportError(f"Missing dependency: {_import_error}")

        prefix = os.path.join(root, f"{pid}_{sess}")

        self.human_traj = Trajectory()
        self.human_traj.loadTrajHDF5(f"{prefix}_human_data.hdf5")
        self.gaze_traj = load_gaze(f"{prefix}_gaze_data.hdf5")

        self.kin = HumanKin()
        if WRIST_JOINT_NAME not in self.kin.inv_index:
            raise KeyError(
                f"'{WRIST_JOINT_NAME}' not in HumanKin.inv_index. "
                f"Available: {list(self.kin.inv_index.keys())}"
            )
        self.wrist_id = self.kin.inv_index[WRIST_JOINT_NAME]

        scene_xml_path = scene_xml or find_scene_xml(root)
        obj_trajs, obj_ids = parseHDF5(f"{prefix}_object_data.hdf5")
        obj_names = parse_object_names(obj_ids, scene_xml_path)

        with h5py.File(f"{prefix}_segmentations.hdf5", "r") as f:
            all_labels = [
                l.decode() if isinstance(l, (bytes, bytearray)) else str(l)
                for l in f["segments"].attrs["labels"]
            ]
            candidate_names = set(l for l in all_labels if l != "null")
            self.segments = []
            for row in f["segments"]:
                start, end, label = row["start"], row["end"], row["label"]
                label = label.decode() if isinstance(label, (bytes, bytearray)) else str(label)
                self.segments.append((int(start), int(end), label))

        self.object_xyz = {}
        for traj, name in zip(obj_trajs, obj_names):
            if name not in candidate_names:
                continue
            data = np.asarray(traj.data)
            self.object_xyz[name] = data[:, :3]

        missing = candidate_names - set(self.object_xyz.keys())
        if missing:
            print(f"  [warn] {pid}_{sess}: no body matched for {missing}")

    def wrist_xyz_at(self, frame_idx):
        self.kin.set_state(self.human_traj, frame_idx)
        return np.array(self.kin.get_position(self.wrist_id), dtype=np.float32)

    def build_trials(self, seq_len=SEQ_LEN, stride=DOWNSAMPLE_STRIDE):
        """
        Build one trial per (null → labeled) segment pair.

        Each trial uses the null approach period immediately before a labeled
        segment. Using the actual null boundary (rather than a fixed lookback)
        prevents frames from the previous labeled segment leaking into the
        window when the null gap is short.
        """
        trials = []
        object_list = sorted(self.object_xyz.keys())
        label_to_idx = {name: i for i, name in enumerate(object_list)}

        for i in range(len(self.segments) - 1):
            null_start, null_end, null_label = self.segments[i]
            next_start, next_end, next_label = self.segments[i + 1]

            if null_label != "null" or next_label not in label_to_idx:
                continue
            # end frames are inclusive, so adjacent segments have next_start == null_end + 1
            if next_start - null_end not in (0, 1):
                continue

            available = list(range(null_start, null_end, stride))
            if len(available) < seq_len:
                continue
            frame_indices = available[-seq_len:]  # last seq_len frames = closest to grasp

            motion = np.asarray(self.human_traj.data)[frame_indices]
            gaze = np.asarray(self.gaze_traj.data)[frame_indices]
            wrist_xyz = np.stack([self.wrist_xyz_at(fi) for fi in frame_indices])
            goal_positions = np.stack(
                [self.object_xyz[name][null_start] for name in object_list]
            )

            trials.append({
                "motion": motion.astype(np.float32),
                "gaze": gaze.astype(np.float32),
                "wrist_xyz": wrist_xyz.astype(np.float32),
                "goal_positions": goal_positions.astype(np.float32),
                "label": label_to_idx[next_label],
            })
        return trials, object_list


class MoGazeDataset(Dataset):
    """
    Aggregates trials across all sessions under `root`.

    LOPO mode (recommended): set `lopo_test_pid` to a participant ID (e.g. 'p1').
    All sessions from that participant go to the test split; all others to train.

    Random mode (legacy): omit `lopo_test_pid`. Trials are split randomly
    across participants, which inflates results.
    """

    def __init__(self, root, split="train", train_frac=0.8, seed=0,
                 scene_xml=None, lopo_test_pid=None):
        if not DEPS_OK:
            raise ImportError(f"Missing dependency: {_import_error}")

        sessions = find_sessions(root)
        if not sessions:
            raise FileNotFoundError(f"No *_human_data.hdf5 files under {root}")

        train_trials, test_trials = [], []
        object_list = None

        for pid, sess in sessions:
            try:
                session = MoGazeSession(root, pid, sess, scene_xml=scene_xml)
                trials, obj_list = session.build_trials()
                object_list = object_list or obj_list
                if lopo_test_pid is not None:
                    (test_trials if pid == lopo_test_pid else train_trials).extend(trials)
                else:
                    train_trials.extend(trials)
            except Exception as e:
                print(f"  [skip {pid}_{sess}] {e}")

        if not train_trials and not test_trials:
            raise RuntimeError("No trials built. Check errors above.")

        if lopo_test_pid is not None:
            chosen = test_trials if split in ("val", "test") else train_trials
            print(f"MoGazeDataset[LOPO test_pid={lopo_test_pid}, {split}]: "
                  f"{len(chosen)} trials, {len(object_list)} objects")
        else:
            rng = np.random.default_rng(seed)
            idx = rng.permutation(len(train_trials))
            cut = int(len(idx) * train_frac)
            chosen_idx = idx[:cut] if split == "train" else idx[cut:]
            chosen = [train_trials[i] for i in chosen_idx]
            print(f"MoGazeDataset[random {split}]: {len(chosen)} trials, "
                  f"{len(object_list)} objects")

        self.trials = chosen
        self.object_list = object_list

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx):
        t = self.trials[idx]
        return {
            "motion": torch.from_numpy(t["motion"]),
            "gaze": torch.from_numpy(t["gaze"]),
            "wrist_xyz": torch.from_numpy(t["wrist_xyz"]),
            "goal_positions": torch.from_numpy(t["goal_positions"]),
            "label": torch.tensor(t["label"], dtype=torch.long),
        }


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "."

    if not DEPS_OK:
        print(f"Missing: {_import_error}")
        sys.exit(1)

    sessions = find_sessions(root)
    print(f"Found {len(sessions)} sessions: {sessions}")
    if not sessions:
        sys.exit(1)

    pid, sess = sessions[0]
    print(f"\nInspecting {pid}_{sess} ...")
    s = MoGazeSession(root, pid, sess)
    print(f"Motion: {np.asarray(s.human_traj.data).shape}")
    print(f"Gaze:   {np.asarray(s.gaze_traj.data).shape}")
    print(f"Objects: {sorted(s.object_xyz.keys())}")
    print(f"Segments: {len(s.segments)}, first 5: {s.segments[:5]}")

    trials, object_list = s.build_trials()
    print(f"\nBuilt {len(trials)} trials")
    if trials:
        for k, v in trials[0].items():
            print(f"  {k}: {np.shape(v) if hasattr(v, 'shape') else v}")
        print(f"\nUse --motion_dim {trials[0]['motion'].shape[-1]} "
              f"--n_goals {len(object_list)}")