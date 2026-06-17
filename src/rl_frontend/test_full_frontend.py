# src/rl_frontend/test_paper_frontend.py

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.rl_frontend.bisst_feature_extractor import (
    BiSSTFeatureMapExtractor,
    image_to_tensor,
)
from src.rl_frontend.visual_projector import RLVisualProjector
from src.rl_frontend.policy_net import RecurrentActorCriticPolicy


def parse_args():
    parser = argparse.ArgumentParser(
        description="Paper-style RL frontend test: image -> frozen G feature map -> trainable visual projector -> policy -> action."
    )

    # BiSST G
    parser.add_argument("--bisst_checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument(
        "--direction",
        type=str,
        default="real2virtual",
        choices=["virtual2real", "real2virtual"],
    )
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--ndf", type=int, default=64)
    parser.add_argument("--n_blocks", type=int, default=9)

    # RL visual projector
    parser.add_argument("--feature_channels", type=int, default=256)
    parser.add_argument("--projector_hidden_channels", type=int, default=128)
    parser.add_argument("--visual_dim", type=int, default=128)
    parser.add_argument("--projector_mlp_hidden_dim", type=int, default=256)

    # Robot state and policy
    parser.add_argument("--robot_state_dim", type=int, default=14)
    parser.add_argument("--fusion_hidden_dim", type=int, default=256)
    parser.add_argument("--fusion_output_dim", type=int, default=256)
    parser.add_argument("--lstm_hidden_dim", type=int, default=512)
    parser.add_argument("--lstm_layers", type=int, default=1)
    parser.add_argument("--action_dim", type=int, default=3)
    parser.add_argument("--action_std_init", type=float, default=0.5)

    parser.add_argument(
        "--robot_state",
        type=float,
        nargs="*",
        default=None,
        help="Optional 14 numbers for robot state. If omitted, use zeros.",
    )

    parser.add_argument(
        "--policy_checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint for paper-style policy. Usually not available yet.",
    )
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--strict", action="store_true")

    return parser.parse_args()


def build_robot_state_tensor(
    robot_state_values,
    robot_state_dim: int,
    device: str,
) -> torch.Tensor:
    if robot_state_values is None or len(robot_state_values) == 0:
        return torch.zeros(
            1,
            robot_state_dim,
            dtype=torch.float32,
            device=device,
        )

    if len(robot_state_values) != robot_state_dim:
        raise ValueError(
            "robot_state length mismatch: expected {}, got {}".format(
                robot_state_dim,
                len(robot_state_values),
            )
        )

    return torch.tensor(
        robot_state_values,
        dtype=torch.float32,
        device=device,
    ).view(1, robot_state_dim)


def load_policy_checkpoint_if_needed(
    policy,
    visual_projector,
    checkpoint_path: Optional[str],
    device: str,
):
    if checkpoint_path is None:
        print("No paper-style policy checkpoint provided. Use randomly initialized RLVisualProjector and policy.")
        return policy, visual_projector

    path = Path(checkpoint_path)

    if not path.exists():
        raise FileNotFoundError("Policy checkpoint does not exist: {}".format(path))

    checkpoint = torch.load(
        path,
        map_location=device,
    )

    if "policy" not in checkpoint:
        raise KeyError("Checkpoint does not contain key 'policy'.")

    if "visual_projector" not in checkpoint:
        raise KeyError("Checkpoint does not contain key 'visual_projector'.")

    visual_projector.load_state_dict(
        checkpoint["visual_projector"],
        strict=True,
    )
    policy.load_state_dict(
        checkpoint["policy"],
        strict=True,
    )

    print("Loaded paper-style policy checkpoint:", checkpoint_path)
    print("Checkpoint epoch:", checkpoint.get("epoch", "unknown"))

    return policy, visual_projector


def main():
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Fall back to CPU.")
        args.device = "cpu"

    print("========== Paper-style RL Frontend Test ==========")
    print("Project root:", PROJECT_ROOT)
    print("BiSST checkpoint:", args.bisst_checkpoint)
    print("Image:", args.image)
    print("Direction:", args.direction)
    print("Feature channels:", args.feature_channels)
    print("Visual dim:", args.visual_dim)
    print("Robot state dim:", args.robot_state_dim)
    print("Action dim:", args.action_dim)
    print("Device:", args.device)

    feature_extractor = BiSSTFeatureMapExtractor(
        checkpoint_path=args.bisst_checkpoint,
        ngf=args.ngf,
        ndf=args.ndf,
        n_blocks=args.n_blocks,
        direction=args.direction,
        device=args.device,
        strict=args.strict,
    )

    visual_projector = RLVisualProjector(
        input_channels=args.feature_channels,
        hidden_channels=args.projector_hidden_channels,
        embedding_dim=args.visual_dim,
        mlp_hidden_dim=args.projector_mlp_hidden_dim,
        normalize=True,
    ).to(args.device)

    policy = RecurrentActorCriticPolicy(
        visual_dim=args.visual_dim,
        sensor_dim=args.robot_state_dim,
        fusion_hidden_dim=args.fusion_hidden_dim,
        fusion_output_dim=args.fusion_output_dim,
        lstm_hidden_dim=args.lstm_hidden_dim,
        lstm_layers=args.lstm_layers,
        action_dim=args.action_dim,
        action_std_init=args.action_std_init,
        dropout=0.0,
    ).to(args.device)

    policy, visual_projector = load_policy_checkpoint_if_needed(
        policy=policy,
        visual_projector=visual_projector,
        checkpoint_path=args.policy_checkpoint,
        device=args.device,
    )

    feature_extractor.eval()
    visual_projector.eval()
    policy.eval()

    image_tensor = image_to_tensor(
        image_path=args.image,
        image_size=args.image_size,
        device=args.device,
    )

    robot_state = build_robot_state_tensor(
        robot_state_values=args.robot_state,
        robot_state_dim=args.robot_state_dim,
        device=args.device,
    )

    with torch.no_grad():
        feature_map = feature_extractor(image_tensor)
        visual_feature = visual_projector(feature_map)

        action, log_prob, value, next_hidden_state = policy.act(
            visual_feature=visual_feature,
            sensor_state=robot_state,
            hidden_state=None,
            deterministic=args.deterministic,
        )

    print("")
    print("========== Outputs ==========")
    print("Image tensor shape:", image_tensor.shape)
    print("Frozen G feature map shape:", feature_map.shape)
    print("RL visual feature shape:", visual_feature.shape)
    print("Robot state shape:", robot_state.shape)
    print("Action shape:", action.shape)
    print("Action [d, rx, ry]:", action.detach().cpu().numpy()[0].tolist())
    print("Value shape:", value.shape)
    print("Value:", value.detach().cpu().numpy()[0].tolist())
    print("Next hidden h shape:", next_hidden_state[0].shape)
    print("Next hidden c shape:", next_hidden_state[1].shape)

    print("")
    print("========== Test passed ==========")


if __name__ == "__main__":
    main()

    #python src/rl_frontend/test_full_frontend.py --bisst_checkpoint checkpoints/bisst/20260617_143913_bisst_real2virtual/latest.pt --image data/processed/real/0001.png --direction real2virtual --feature_channels 256 --robot_state_dim 14 --deterministic