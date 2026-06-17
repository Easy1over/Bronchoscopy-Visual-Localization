# src/rl_frontend/train_bc.py

import argparse
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.rl_frontend.dataset import (
    DemonstrationSequenceDataset,
    collate_demonstration_batch,
)
from src.rl_frontend.bisst_feature_extractor import BiSSTFeatureMapExtractor
from src.rl_frontend.visual_projector import RLVisualProjector
from src.rl_frontend.policy_net import RecurrentActorCriticPolicy


IMAGE_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Paper-style Behavior Cloning training for RL frontend."
    )

    # Data
    parser.add_argument(
        "--data_mode",
        type=str,
        default="paper_image",
        choices=["paper_image", "feature_npz"],
        help=(
            "paper_image: image -> frozen BiSST G -> feature map -> trainable RLVisualProjector -> policy. "
            "feature_npz: old debug mode, precomputed 128-d visual_features -> policy."
        ),
    )
    parser.add_argument(
        "--demo_path",
        type=str,
        default=str(PROJECT_ROOT / "data" / "rl" / "demonstrations" / "dummy_image_demo.npz"),
        help="Path to demonstration npz file.",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=8,
        help="Sequence length for LSTM behavior cloning.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Sliding window stride.",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Image size for paper_image mode.",
    )

    # Dummy image demo for testing paper_image mode
    parser.add_argument(
        "--create_dummy_image_demo",
        action="store_true",
        help="Create a dummy image-path demonstration npz for testing paper_image mode.",
    )
    parser.add_argument(
        "--dummy_image_dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "processed" / "real"),
        help="Image directory used to create dummy image demo.",
    )
    parser.add_argument("--num_episodes", type=int, default=3)
    parser.add_argument("--episode_length", type=int, default=30)

    # Frozen BiSST G for paper_image mode
    parser.add_argument(
        "--bisst_checkpoint",
        type=str,
        default=None,
        help="Path to trained BiSST checkpoint. Required for paper_image mode.",
    )
    parser.add_argument(
        "--direction",
        type=str,
        default="real2virtual",
        choices=["virtual2real", "real2virtual"],
        help="BiSST direction used to construct frozen G.",
    )
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--ndf", type=int, default=64)
    parser.add_argument("--n_blocks", type=int, default=9)
    parser.add_argument("--strict_bisst", action="store_true")

    # Trainable RL visual projector
    parser.add_argument(
        "--feature_channels",
        type=int,
        default=256,
        help="Channel number of frozen G feature map S. Check with bisst_feature_extractor.py first.",
    )
    parser.add_argument(
        "--projector_hidden_channels",
        type=int,
        default=128,
        help="CNN hidden channels in RLVisualProjector.",
    )
    parser.add_argument(
        "--projector_mlp_hidden_dim",
        type=int,
        default=256,
        help="MLP hidden dimension in RLVisualProjector.",
    )

    # Policy
    parser.add_argument(
        "--visual_dim",
        type=int,
        default=128,
        help="Dimension of RL visual vector.",
    )
    parser.add_argument(
        "--sensor_dim",
        type=int,
        default=14,
        help="Dimension of robot low-dimensional state.",
    )
    parser.add_argument(
        "--action_dim",
        type=int,
        default=3,
        help="Action dimension: [d, rx, ry].",
    )
    parser.add_argument("--fusion_hidden_dim", type=int, default=256)
    parser.add_argument("--fusion_output_dim", type=int, default=256)
    parser.add_argument("--lstm_hidden_dim", type=int, default=512)
    parser.add_argument("--lstm_layers", type=int, default=1)
    parser.add_argument("--action_std_init", type=float, default=0.5)

    # Training
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=1.0,
        help="Gradient clipping max norm. Use <=0 to disable.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Use 0 first on Windows.",
    )
    parser.add_argument("--print_freq", type=int, default=10)

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "checkpoints" / "rl_frontend" / "bc_policy_paper"),
    )
    parser.add_argument("--save_freq", type=int, default=5)

    # Device
    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def collect_image_paths(image_dir: str) -> List[Path]:
    root = Path(image_dir)

    if not root.exists():
        raise FileNotFoundError("Image directory does not exist: {}".format(root))

    image_paths = []

    for path in root.rglob("*"):
        if path.is_file() and is_image_file(path):
            image_paths.append(path)

    image_paths = sorted(image_paths)

    if len(image_paths) == 0:
        raise RuntimeError("No images found in: {}".format(root))

    return image_paths


def image_to_tensor_from_path(
    image_path: str,
    image_size: int,
) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    image = image.resize((image_size, image_size), Image.BICUBIC)

    array = np.array(image, dtype=np.uint8)
    tensor = torch.from_numpy(array).permute(2, 0, 1).float() / 255.0
    tensor = tensor * 2.0 - 1.0

    return tensor


def create_dummy_image_demonstration_npz(
    save_path: str,
    image_dir: str,
    num_episodes: int,
    episode_length: int,
    sensor_dim: int,
    action_dim: int,
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    image_paths = collect_image_paths(image_dir)

    total_count = num_episodes * episode_length

    rng = np.random.RandomState(1234)

    saved_image_paths = []
    sensor_states = []
    actions = []
    episode_ids = []
    timesteps = []

    for episode_id in range(num_episodes):
        for t in range(episode_length):
            global_index = episode_id * episode_length + t
            image_path = image_paths[global_index % len(image_paths)]

            saved_image_paths.append(str(image_path))
            sensor_states.append(rng.randn(sensor_dim).astype(np.float32))
            actions.append(
                rng.uniform(
                    low=-1.0,
                    high=1.0,
                    size=(action_dim,),
                ).astype(np.float32)
            )
            episode_ids.append(episode_id)
            timesteps.append(t)

    np.savez(
        save_path,
        image_paths=np.array(saved_image_paths),
        sensor_states=np.stack(sensor_states, axis=0).astype(np.float32),
        actions=np.stack(actions, axis=0).astype(np.float32),
        episode_ids=np.array(episode_ids, dtype=np.int64),
        timesteps=np.array(timesteps, dtype=np.int64),
    )

    print("Created dummy image demonstration:", save_path)
    print("image_paths:", total_count)
    print("sensor_states:", (total_count, sensor_dim))
    print("actions:", (total_count, action_dim))


class ImageDemonstrationSequenceDataset(Dataset):
    """
    Paper-style demonstration dataset.

    Expected npz format:
        image_paths: [N]
        sensor_states: [N, sensor_dim]
        actions: [N, action_dim]
        episode_ids: [N]
        timesteps: [N]

    Output:
        image_seq: [T, 3, H, W]
        sensor_state_seq: [T, sensor_dim]
        action_seq: [T, action_dim]
        valid_mask: [T]
    """

    def __init__(
        self,
        demo_path: str,
        seq_len: int,
        stride: int,
        image_size: int,
    ) -> None:
        super().__init__()

        if seq_len < 1:
            raise ValueError("seq_len must be >= 1.")

        if stride < 1:
            raise ValueError("stride must be >= 1.")

        self.demo_path = Path(demo_path)
        self.seq_len = seq_len
        self.stride = stride
        self.image_size = image_size

        if not self.demo_path.exists():
            raise FileNotFoundError(
                "Demonstration file does not exist: {}".format(self.demo_path)
            )

        loaded = np.load(
            self.demo_path,
            allow_pickle=False,
        )

        required_keys = [
            "image_paths",
            "sensor_states",
            "actions",
            "episode_ids",
            "timesteps",
        ]

        for key in required_keys:
            if key not in loaded.files:
                raise KeyError(
                    "Image demonstration npz does not contain key '{}'. "
                    "Required keys: {}".format(key, required_keys)
                )

        self.image_paths = loaded["image_paths"].astype(str)
        self.sensor_states = loaded["sensor_states"].astype(np.float32)
        self.actions = loaded["actions"].astype(np.float32)
        self.episode_ids = loaded["episode_ids"].astype(np.int64)
        self.timesteps = loaded["timesteps"].astype(np.int64)

        self._validate_shapes()
        self.windows = self._build_windows()

        if len(self.windows) == 0:
            raise RuntimeError(
                "No valid sequence windows were created. "
                "Try reducing seq_len or check episode lengths."
            )

    def _validate_shapes(self) -> None:
        n = len(self.image_paths)

        if self.sensor_states.shape[0] != n:
            raise ValueError("sensor_states length does not match image_paths.")

        if self.actions.shape[0] != n:
            raise ValueError("actions length does not match image_paths.")

        if self.episode_ids.shape[0] != n:
            raise ValueError("episode_ids length does not match image_paths.")

        if self.timesteps.shape[0] != n:
            raise ValueError("timesteps length does not match image_paths.")

        if self.sensor_states.ndim != 2:
            raise ValueError("sensor_states should be [N, sensor_dim].")

        if self.actions.ndim != 2:
            raise ValueError("actions should be [N, action_dim].")

    def _build_windows(self):
        windows = []
        unique_episode_ids = np.unique(self.episode_ids)

        for episode_id in unique_episode_ids:
            indices = np.where(self.episode_ids == episode_id)[0]

            if len(indices) == 0:
                continue

            sorted_order = np.argsort(self.timesteps[indices])
            indices = indices[sorted_order]

            episode_len = len(indices)

            if episode_len < self.seq_len:
                continue

            start_local = 0
            while start_local + self.seq_len <= episode_len:
                window_indices = indices[start_local:start_local + self.seq_len]
                windows.append(window_indices.astype(np.int64))
                start_local += self.stride

        return windows

    def __len__(self):
        return len(self.windows)

    @property
    def sensor_dim(self) -> int:
        return int(self.sensor_states.shape[1])

    @property
    def action_dim(self) -> int:
        return int(self.actions.shape[1])

    def __getitem__(self, index):
        window_indices = self.windows[index]

        image_tensors = []
        for idx in window_indices:
            image_tensors.append(
                image_to_tensor_from_path(
                    image_path=self.image_paths[int(idx)],
                    image_size=self.image_size,
                )
            )

        image_seq = torch.stack(image_tensors, dim=0)

        sensor_state_seq = torch.from_numpy(
            self.sensor_states[window_indices]
        )
        action_seq = torch.from_numpy(
            self.actions[window_indices]
        )

        valid_mask = torch.ones(
            self.seq_len,
            dtype=torch.float32,
        )

        return {
            "image_seq": image_seq,
            "sensor_state_seq": sensor_state_seq,
            "action_seq": action_seq,
            "valid_mask": valid_mask,
        }


def collate_image_demonstration_batch(batch):
    image_seq = torch.stack(
        [item["image_seq"] for item in batch],
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
        "image_seq": image_seq,
        "sensor_state_seq": sensor_state_seq,
        "action_seq": action_seq,
        "valid_mask": valid_mask,
    }


def create_policy(args) -> RecurrentActorCriticPolicy:
    policy = RecurrentActorCriticPolicy(
        visual_dim=args.visual_dim,
        sensor_dim=args.sensor_dim,
        fusion_hidden_dim=args.fusion_hidden_dim,
        fusion_output_dim=args.fusion_output_dim,
        lstm_hidden_dim=args.lstm_hidden_dim,
        lstm_layers=args.lstm_layers,
        action_dim=args.action_dim,
        action_std_init=args.action_std_init,
        dropout=0.0,
    )

    return policy


def create_visual_projector(args) -> RLVisualProjector:
    visual_projector = RLVisualProjector(
        input_channels=args.feature_channels,
        hidden_channels=args.projector_hidden_channels,
        embedding_dim=args.visual_dim,
        mlp_hidden_dim=args.projector_mlp_hidden_dim,
        normalize=True,
    )

    return visual_projector


def masked_mse_loss(
    pred_action: torch.Tensor,
    target_action: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    if pred_action.shape != target_action.shape:
        raise ValueError(
            "pred_action and target_action shape mismatch: {} vs {}".format(
                tuple(pred_action.shape),
                tuple(target_action.shape),
            )
        )

    if valid_mask.dim() != 2:
        raise ValueError(
            "valid_mask should have shape [B, T], got {}".format(
                tuple(valid_mask.shape)
            )
        )

    diff = pred_action - target_action
    loss_per_step = torch.mean(
        diff * diff,
        dim=-1,
    )

    valid_mask = valid_mask.float()
    masked_loss = loss_per_step * valid_mask

    denom = torch.clamp(
        valid_mask.sum(),
        min=1.0,
    )

    loss = masked_loss.sum() / denom

    return loss


def extract_visual_sequence_from_images(
    image_seq: torch.Tensor,
    feature_extractor: BiSSTFeatureMapExtractor,
    visual_projector: RLVisualProjector,
    args,
) -> torch.Tensor:
    """
    Args:
        image_seq: [B, T, 3, H, W]

    Returns:
        visual_feature_seq: [B, T, visual_dim]
    """
    if image_seq.dim() != 5:
        raise ValueError(
            "image_seq should be [B, T, 3, H, W], got {}".format(
                tuple(image_seq.shape)
            )
        )

    batch_size = image_seq.shape[0]
    seq_len = image_seq.shape[1]

    image_flat = image_seq.reshape(
        batch_size * seq_len,
        image_seq.shape[2],
        image_seq.shape[3],
        image_seq.shape[4],
    ).to(args.device)

    # Frozen G feature extraction. No gradient should flow into G.
    with torch.no_grad():
        feature_map_flat = feature_extractor(image_flat)

    # RLVisualProjector is trainable, so no no_grad here.
    visual_feature_flat = visual_projector(feature_map_flat)

    visual_feature_seq = visual_feature_flat.reshape(
        batch_size,
        seq_len,
        args.visual_dim,
    )

    return visual_feature_seq


def forward_batch(
    batch,
    policy,
    args,
    feature_extractor=None,
    visual_projector=None,
):
    sensor_state_seq = batch["sensor_state_seq"].to(args.device)
    action_seq = batch["action_seq"].to(args.device)
    valid_mask = batch["valid_mask"].to(args.device)

    if args.data_mode == "paper_image":
        image_seq = batch["image_seq"].to(args.device)

        visual_feature_seq = extract_visual_sequence_from_images(
            image_seq=image_seq,
            feature_extractor=feature_extractor,
            visual_projector=visual_projector,
            args=args,
        )

    elif args.data_mode == "feature_npz":
        visual_feature_seq = batch["visual_feature_seq"].to(args.device)

    else:
        raise ValueError("Unsupported data_mode: {}".format(args.data_mode))

    action_mean, action_std, value, _ = policy(
        visual_feature_seq=visual_feature_seq,
        sensor_state_seq=sensor_state_seq,
        hidden_state=None,
    )

    loss = masked_mse_loss(
        pred_action=action_mean,
        target_action=action_seq,
        valid_mask=valid_mask,
    )

    return loss, visual_feature_seq, action_mean, value


def train_one_epoch(
    policy,
    dataloader,
    optimizer,
    args,
    epoch,
    feature_extractor=None,
    visual_projector=None,
):
    policy.train()

    if visual_projector is not None:
        visual_projector.train()

    if feature_extractor is not None:
        feature_extractor.eval()

    total_loss = 0.0
    total_count = 0

    start_time = time.time()

    for iteration, batch in enumerate(dataloader, start=1):
        loss, visual_feature_seq, action_mean, value = forward_batch(
            batch=batch,
            policy=policy,
            args=args,
            feature_extractor=feature_extractor,
            visual_projector=visual_projector,
        )

        optimizer.zero_grad()
        loss.backward()

        if args.grad_clip > 0:
            params = list(policy.parameters())

            if visual_projector is not None:
                params += list(visual_projector.parameters())

            nn.utils.clip_grad_norm_(
                params,
                max_norm=args.grad_clip,
            )

        optimizer.step()

        batch_size = visual_feature_seq.shape[0]
        total_loss += loss.item() * batch_size
        total_count += batch_size

        if iteration == 1 or iteration % args.print_freq == 0:
            elapsed = time.time() - start_time
            print(
                "Epoch [{}/{}] Iter [{}] loss: {:.6f} elapsed: {:.1f}s".format(
                    epoch,
                    args.epochs,
                    iteration,
                    loss.item(),
                    elapsed,
                )
            )

    mean_loss = total_loss / max(total_count, 1)

    return mean_loss


def validate(
    policy,
    dataloader,
    args,
    feature_extractor=None,
    visual_projector=None,
):
    policy.eval()

    if visual_projector is not None:
        visual_projector.eval()

    if feature_extractor is not None:
        feature_extractor.eval()

    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for batch in dataloader:
            loss, visual_feature_seq, action_mean, value = forward_batch(
                batch=batch,
                policy=policy,
                args=args,
                feature_extractor=feature_extractor,
                visual_projector=visual_projector,
            )

            batch_size = visual_feature_seq.shape[0]
            total_loss += loss.item() * batch_size
            total_count += batch_size

    mean_loss = total_loss / max(total_count, 1)

    return mean_loss


def save_checkpoint(
    save_path: Path,
    policy,
    optimizer,
    epoch,
    train_loss,
    val_loss,
    args,
    visual_projector=None,
):
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "policy": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "args": vars(args),
    }

    if visual_projector is not None:
        checkpoint["visual_projector"] = visual_projector.state_dict()

    torch.save(
        checkpoint,
        save_path,
    )


def split_dataset(
    dataset,
    val_ratio,
):
    dataset_size = len(dataset)

    if dataset_size < 2 or val_ratio <= 0:
        return dataset, None

    val_size = int(dataset_size * val_ratio)
    val_size = max(val_size, 1)
    train_size = dataset_size - val_size

    if train_size < 1:
        train_size = dataset_size
        val_size = 0

    if val_size == 0:
        return dataset, None

    generator = torch.Generator().manual_seed(1234)

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
    )

    return train_dataset, val_dataset


def build_dataset_and_collate(args):
    if args.data_mode == "paper_image":
        dataset = ImageDemonstrationSequenceDataset(
            demo_path=args.demo_path,
            seq_len=args.seq_len,
            stride=args.stride,
            image_size=args.image_size,
        )

        collate_fn = collate_image_demonstration_batch

        if dataset.sensor_dim != args.sensor_dim:
            raise ValueError(
                "sensor_dim mismatch: dataset has {}, args has {}".format(
                    dataset.sensor_dim,
                    args.sensor_dim,
                )
            )

        if dataset.action_dim != args.action_dim:
            raise ValueError(
                "action_dim mismatch: dataset has {}, args has {}".format(
                    dataset.action_dim,
                    args.action_dim,
                )
            )

    elif args.data_mode == "feature_npz":
        dataset = DemonstrationSequenceDataset(
            demo_path=args.demo_path,
            seq_len=args.seq_len,
            stride=args.stride,
        )

        collate_fn = collate_demonstration_batch

        if dataset.visual_dim != args.visual_dim:
            raise ValueError(
                "visual_dim mismatch: dataset has {}, args has {}".format(
                    dataset.visual_dim,
                    args.visual_dim,
                )
            )

        if dataset.sensor_dim != args.sensor_dim:
            raise ValueError(
                "sensor_dim mismatch: dataset has {}, args has {}".format(
                    dataset.sensor_dim,
                    args.sensor_dim,
                )
            )

        if dataset.action_dim != args.action_dim:
            raise ValueError(
                "action_dim mismatch: dataset has {}, args has {}".format(
                    dataset.action_dim,
                    args.action_dim,
                )
            )

    else:
        raise ValueError("Unsupported data_mode: {}".format(args.data_mode))

    return dataset, collate_fn


def main():
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Fall back to CPU.")
        args.device = "cpu"

    if args.epochs < 1:
        raise ValueError("epochs must be >= 1.")

    if args.batch_size < 1:
        raise ValueError("batch_size must be >= 1.")

    if args.print_freq < 1:
        raise ValueError("print_freq must be >= 1.")

    if args.data_mode == "paper_image" and args.bisst_checkpoint is None:
        if not args.create_dummy_image_demo:
            raise ValueError("paper_image mode requires --bisst_checkpoint.")

    if args.create_dummy_image_demo:
        create_dummy_image_demonstration_npz(
            save_path=args.demo_path,
            image_dir=args.dummy_image_dir,
            num_episodes=args.num_episodes,
            episode_length=args.episode_length,
            sensor_dim=args.sensor_dim,
            action_dim=args.action_dim,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset, collate_fn = build_dataset_and_collate(args)

    train_dataset, val_dataset = split_dataset(
        dataset=dataset,
        val_ratio=args.val_ratio,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=collate_fn,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False,
            collate_fn=collate_fn,
        )

    policy = create_policy(args).to(args.device)

    feature_extractor = None
    visual_projector = None

    if args.data_mode == "paper_image":
        feature_extractor = BiSSTFeatureMapExtractor(
            checkpoint_path=args.bisst_checkpoint,
            ngf=args.ngf,
            ndf=args.ndf,
            n_blocks=args.n_blocks,
            direction=args.direction,
            device=args.device,
            strict=args.strict_bisst,
        )

        visual_projector = create_visual_projector(args).to(args.device)

        optimizer_params = list(policy.parameters()) + list(visual_projector.parameters())

    else:
        optimizer_params = list(policy.parameters())

    optimizer = torch.optim.AdamW(
        optimizer_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_loss = None

    print("========== Behavior Cloning Training ==========")
    print("Project root:", PROJECT_ROOT)
    print("Data mode:", args.data_mode)
    print("Demo path:", args.demo_path)
    print("Output dir:", output_dir)
    print("Dataset size:", len(dataset))
    print("Train size:", len(train_dataset))
    print("Val size:", len(val_dataset) if val_dataset is not None else 0)
    print("Seq len:", args.seq_len)
    print("Image size:", args.image_size)
    print("Visual dim:", args.visual_dim)
    print("Sensor dim:", args.sensor_dim)
    print("Action dim:", args.action_dim)
    print("LSTM hidden dim:", args.lstm_hidden_dim)
    print("Device:", args.device)

    if args.data_mode == "paper_image":
        print("BiSST checkpoint:", args.bisst_checkpoint)
        print("Direction:", args.direction)
        print("Frozen G feature channels:", args.feature_channels)
        print("Trainable RLVisualProjector: True")
    else:
        print("Trainable RLVisualProjector: False, using precomputed visual_features.")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            policy=policy,
            dataloader=train_loader,
            optimizer=optimizer,
            args=args,
            epoch=epoch,
            feature_extractor=feature_extractor,
            visual_projector=visual_projector,
        )

        if val_loader is not None:
            val_loss = validate(
                policy=policy,
                dataloader=val_loader,
                args=args,
                feature_extractor=feature_extractor,
                visual_projector=visual_projector,
            )
        else:
            val_loss = train_loss

        print(
            "Epoch [{}/{}] train_loss: {:.6f} val_loss: {:.6f}".format(
                epoch,
                args.epochs,
                train_loss,
                val_loss,
            )
        )

        latest_path = output_dir / "latest.pt"
        save_checkpoint(
            save_path=latest_path,
            policy=policy,
            optimizer=optimizer,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            args=args,
            visual_projector=visual_projector,
        )

        if epoch % args.save_freq == 0 or epoch == args.epochs:
            epoch_path = output_dir / "epoch_{:04d}.pt".format(epoch)
            save_checkpoint(
                save_path=epoch_path,
                policy=policy,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                args=args,
                visual_projector=visual_projector,
            )

        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = output_dir / "best.pt"
            save_checkpoint(
                save_path=best_path,
                policy=policy,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                args=args,
                visual_projector=visual_projector,
            )

    print("")
    print("========== BC training finished ==========")
    print("Latest checkpoint:", output_dir / "latest.pt")
    print("Best checkpoint:", output_dir / "best.pt")

#python src/rl_frontend/test_full_frontend.py --bisst_checkpoint checkpoints/bisst/20260617_143913_bisst_real2virtual/latest.pt --image data/processed/real/0001.png --direction real2virtual --feature_channels 256 --robot_state_dim 14 --policy_checkpoint checkpoints/rl_frontend/bc_policy_paper_dummy/latest.pt --deterministic
if __name__ == "__main__":
    main()