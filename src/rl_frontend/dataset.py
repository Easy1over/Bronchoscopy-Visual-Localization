# src/rl_frontend/dataset.py

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))


REQUIRED_KEYS = [
    "visual_features",
    "sensor_states",
    "actions",
    "episode_ids",
    "timesteps",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test demonstration dataset for BC / RL frontend."
    )

    parser.add_argument(
        "--demo_path",
        type=str,
        default=str(PROJECT_ROOT / "data" / "rl" / "demonstrations" / "dummy_demo.npz"),
        help="Path to demonstration npz file.",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=8,
        help="Sequence length for LSTM training.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Sliding window stride.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for dataset test.",
    )
    parser.add_argument(
        "--create_dummy",
        action="store_true",
        help="Create a dummy demonstration npz for testing.",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=3,
        help="Number of dummy episodes.",
    )
    parser.add_argument(
        "--episode_length",
        type=int,
        default=30,
        help="Length of each dummy episode.",
    )
    parser.add_argument(
        "--visual_dim",
        type=int,
        default=128,
        help="Visual feature dimension.",
    )
    parser.add_argument(
        "--sensor_dim",
        type=int,
        default=9,
        help="Sensor state dimension.",
    )
    parser.add_argument(
        "--action_dim",
        type=int,
        default=3,
        help="Action dimension. Paper-style action is [d, rx, ry].",
    )

    return parser.parse_args()


def check_npz_keys(data: Dict[str, np.ndarray]) -> None:
    for key in REQUIRED_KEYS:
        if key not in data:
            raise KeyError(
                "Demonstration npz does not contain key '{}'. Required keys: {}".format(
                    key,
                    REQUIRED_KEYS,
                )
            )


def create_dummy_demonstration_npz(
    save_path: str,
    num_episodes: int = 3,
    episode_length: int = 30,
    visual_dim: int = 128,
    sensor_dim: int = 14,
    action_dim: int = 3,
) -> None:
    """
    Create a dummy demonstration file for testing.

    Format:
        visual_features: [N, visual_dim]
        sensor_states: [N, sensor_dim]
        actions: [N, action_dim]
        episode_ids: [N]
        timesteps: [N]
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    visual_features = []
    sensor_states = []
    actions = []
    episode_ids = []
    timesteps = []

    rng = np.random.RandomState(1234)

    for episode_id in range(num_episodes):
        for t in range(episode_length):
            visual_feature = rng.randn(visual_dim).astype(np.float32)
            sensor_state = rng.randn(sensor_dim).astype(np.float32)

            # Paper-style continuous action: [d, rx, ry], normalized to [-1, 1].
            action = rng.uniform(
                low=-1.0,
                high=1.0,
                size=(action_dim,),
            ).astype(np.float32)

            visual_features.append(visual_feature)
            sensor_states.append(sensor_state)
            actions.append(action)
            episode_ids.append(episode_id)
            timesteps.append(t)

    visual_features = np.stack(visual_features, axis=0).astype(np.float32)
    sensor_states = np.stack(sensor_states, axis=0).astype(np.float32)
    actions = np.stack(actions, axis=0).astype(np.float32)
    episode_ids = np.array(episode_ids, dtype=np.int64)
    timesteps = np.array(timesteps, dtype=np.int64)

    np.savez(
        save_path,
        visual_features=visual_features,
        sensor_states=sensor_states,
        actions=actions,
        episode_ids=episode_ids,
        timesteps=timesteps,
    )

    print("Created dummy demonstration:", save_path)
    print("visual_features:", visual_features.shape)
    print("sensor_states:", sensor_states.shape)
    print("actions:", actions.shape)
    print("episode_ids:", episode_ids.shape)
    print("timesteps:", timesteps.shape)


class DemonstrationSequenceDataset(Dataset):
    """
    Demonstration sequence dataset for imitation learning / behavior cloning.

    Expected npz format:
        visual_features: [N, visual_dim]
        sensor_states: [N, sensor_dim]
        actions: [N, action_dim]
        episode_ids: [N]
        timesteps: [N]

    Output sample:
        visual_feature_seq: [seq_len, visual_dim]
        sensor_state_seq: [seq_len, sensor_dim]
        action_seq: [seq_len, action_dim]
        valid_mask: [seq_len]

    Current version only builds fixed-length windows inside each episode.
    It does not cross episode boundaries.
    """

    def __init__(
        self,
        demo_path: str,
        seq_len: int = 8,
        stride: int = 1,
    ) -> None:
        super().__init__()

        if seq_len < 1:
            raise ValueError("seq_len must be >= 1.")

        if stride < 1:
            raise ValueError("stride must be >= 1.")

        self.demo_path = Path(demo_path)
        self.seq_len = seq_len
        self.stride = stride

        if not self.demo_path.exists():
            raise FileNotFoundError(
                "Demonstration file does not exist: {}".format(self.demo_path)
            )

        loaded = np.load(
            self.demo_path,
            allow_pickle=False,
        )

        data = {}
        for key in loaded.files:
            data[key] = loaded[key]

        check_npz_keys(data)

        self.visual_features = data["visual_features"].astype(np.float32)
        self.sensor_states = data["sensor_states"].astype(np.float32)
        self.actions = data["actions"].astype(np.float32)
        self.episode_ids = data["episode_ids"].astype(np.int64)
        self.timesteps = data["timesteps"].astype(np.int64)

        self._validate_shapes()
        self.windows = self._build_windows()

        if len(self.windows) == 0:
            raise RuntimeError(
                "No valid sequence windows were created. "
                "Try reducing seq_len or check episode lengths."
            )

    def _validate_shapes(self) -> None:
        n = self.visual_features.shape[0]

        if self.sensor_states.shape[0] != n:
            raise ValueError("sensor_states length does not match visual_features.")

        if self.actions.shape[0] != n:
            raise ValueError("actions length does not match visual_features.")

        if self.episode_ids.shape[0] != n:
            raise ValueError("episode_ids length does not match visual_features.")

        if self.timesteps.shape[0] != n:
            raise ValueError("timesteps length does not match visual_features.")

        if self.visual_features.ndim != 2:
            raise ValueError(
                "visual_features should be [N, visual_dim], got {}".format(
                    self.visual_features.shape
                )
            )

        if self.sensor_states.ndim != 2:
            raise ValueError(
                "sensor_states should be [N, sensor_dim], got {}".format(
                    self.sensor_states.shape
                )
            )

        if self.actions.ndim != 2:
            raise ValueError(
                "actions should be [N, action_dim], got {}".format(
                    self.actions.shape
                )
            )

    def _build_windows(self) -> List[Tuple[int, int]]:
        """
        Build fixed-length sequence windows.

        Returns:
            list of (start_index, end_index), where end_index is exclusive.
        """
        windows = []

        unique_episode_ids = np.unique(self.episode_ids)

        for episode_id in unique_episode_ids:
            indices = np.where(self.episode_ids == episode_id)[0]

            if len(indices) == 0:
                continue

            # Sort by timestep within this episode.
            sorted_order = np.argsort(self.timesteps[indices])
            indices = indices[sorted_order]

            episode_len = len(indices)

            if episode_len < self.seq_len:
                continue

            start_local = 0
            while start_local + self.seq_len <= episode_len:
                window_indices = indices[start_local:start_local + self.seq_len]

                # Store absolute start/end only if the underlying indices are contiguous
                # in the sorted episode list. We keep explicit windows using first/last.
                windows.append(
                    (
                        int(window_indices[0]),
                        int(window_indices[-1]) + 1,
                    )
                )

                start_local += self.stride

        return windows

    def __len__(self):
        return len(self.windows)

    @property
    def visual_dim(self) -> int:
        return int(self.visual_features.shape[1])

    @property
    def sensor_dim(self) -> int:
        return int(self.sensor_states.shape[1])

    @property
    def action_dim(self) -> int:
        return int(self.actions.shape[1])

    def __getitem__(self, index):
        start_idx, end_idx = self.windows[index]

        visual_feature_seq = self.visual_features[start_idx:end_idx]
        sensor_state_seq = self.sensor_states[start_idx:end_idx]
        action_seq = self.actions[start_idx:end_idx]

        if visual_feature_seq.shape[0] != self.seq_len:
            raise RuntimeError(
                "Invalid window length: expected {}, got {}".format(
                    self.seq_len,
                    visual_feature_seq.shape[0],
                )
            )

        valid_mask = np.ones(
            (self.seq_len,),
            dtype=np.float32,
        )

        sample = {
            "visual_feature_seq": torch.from_numpy(visual_feature_seq),
            "sensor_state_seq": torch.from_numpy(sensor_state_seq),
            "action_seq": torch.from_numpy(action_seq),
            "valid_mask": torch.from_numpy(valid_mask),
        }

        return sample


def collate_demonstration_batch(batch):
    visual_feature_seq = torch.stack(
        [item["visual_feature_seq"] for item in batch],
        dim=0,
    )

    sensor_state_seq = torch.stack(
        [item["sensor_state_seq"] for item in batch],
        dim=0,
    )

    action_seq = torch.stack(
        [item["action_seq"] for item in batch],
        dim=0,
    )

    valid_mask = torch.stack(
        [item["valid_mask"] for item in batch],
        dim=0,
    )

    return {
        "visual_feature_seq": visual_feature_seq,
        "sensor_state_seq": sensor_state_seq,
        "action_seq": action_seq,
        "valid_mask": valid_mask,
    }


def main():
    args = parse_args()

    if args.create_dummy:
        create_dummy_demonstration_npz(
            save_path=args.demo_path,
            num_episodes=args.num_episodes,
            episode_length=args.episode_length,
            visual_dim=args.visual_dim,
            sensor_dim=args.sensor_dim,
            action_dim=args.action_dim,
        )

    dataset = DemonstrationSequenceDataset(
        demo_path=args.demo_path,
        seq_len=args.seq_len,
        stride=args.stride,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
        collate_fn=collate_demonstration_batch,
    )

    print("========== Demonstration Dataset Test ==========")
    print("Demo path:", args.demo_path)
    print("Dataset size:", len(dataset))
    print("Visual dim:", dataset.visual_dim)
    print("Sensor dim:", dataset.sensor_dim)
    print("Action dim:", dataset.action_dim)
    print("Seq len:", args.seq_len)
    print("Stride:", args.stride)

    batch = next(iter(dataloader))

    print("")
    print("Batch visual_feature_seq:", batch["visual_feature_seq"].shape)
    print("Batch sensor_state_seq:", batch["sensor_state_seq"].shape)
    print("Batch action_seq:", batch["action_seq"].shape)
    print("Batch valid_mask:", batch["valid_mask"].shape)


if __name__ == "__main__":
    main()