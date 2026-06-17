# src/rl_frontend/visual_projector.py

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))


class RLVisualProjector(nn.Module):
    """
    Trainable RL visual projector.

    This module is different from BiSST Eh.

    BiSST Eh:
        used for structural semantic consistency during BiSST training.

    RLVisualProjector:
        used for downstream policy learning.
        It receives frozen BiSST G feature map S and produces a compact
        visual vector for control.

    Pipeline:
        feature_map S [B, C, H, W]
            -> CNN
            -> global pooling
            -> MLP
            -> visual_embedding [B, embedding_dim]
    """

    def __init__(
        self,
        input_channels: int = 256,
        hidden_channels: int = 128,
        embedding_dim: int = 128,
        mlp_hidden_dim: int = 256,
        normalize: bool = True,
    ) -> None:
        super().__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.embedding_dim = embedding_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        self.normalize = normalize

        self.cnn = nn.Sequential(
            nn.Conv2d(
                input_channels,
                hidden_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.InstanceNorm2d(
                hidden_channels,
                affine=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.InstanceNorm2d(
                hidden_channels,
                affine=True,
            ),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, mlp_hidden_dim),
            nn.LayerNorm(mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_dim, embedding_dim),
        )

    def forward(
        self,
        feature_map: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            feature_map: [B, C, H, W]

        Returns:
            visual_embedding: [B, embedding_dim]
        """
        if feature_map.dim() != 4:
            raise ValueError(
                "feature_map should be [B, C, H, W], got {}".format(
                    tuple(feature_map.shape)
                )
            )

        if feature_map.shape[1] != self.input_channels:
            raise ValueError(
                "feature_map channel mismatch: expected {}, got {}".format(
                    self.input_channels,
                    feature_map.shape[1],
                )
            )

        x = self.cnn(feature_map)
        x = torch.flatten(x, start_dim=1)
        visual_embedding = self.mlp(x)

        if self.normalize:
            visual_embedding = F.normalize(
                visual_embedding,
                p=2,
                dim=1,
            )

        return visual_embedding


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test RL visual projector."
    )

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--input_channels", type=int, default=256)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--mlp_hidden_dim", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Fall back to CPU.")
        args.device = "cpu"

    projector = RLVisualProjector(
        input_channels=args.input_channels,
        hidden_channels=args.hidden_channels,
        embedding_dim=args.embedding_dim,
        mlp_hidden_dim=args.mlp_hidden_dim,
        normalize=True,
    ).to(args.device)

    feature_map = torch.randn(
        args.batch_size,
        args.input_channels,
        args.height,
        args.width,
        device=args.device,
    )

    visual_embedding = projector(feature_map)

    print("========== RL Visual Projector Test ==========")
    print("Feature map shape:", feature_map.shape)
    print("Visual embedding shape:", visual_embedding.shape)
    print("Visual embedding dtype:", visual_embedding.dtype)
    print("Visual embedding device:", visual_embedding.device)
    print("Visual embedding norm:", torch.norm(visual_embedding, dim=1))


if __name__ == "__main__":
    main()