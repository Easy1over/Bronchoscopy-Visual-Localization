# src/rl_frontend/state_encoder.py

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test multimodal state encoder for RL frontend."
    )

    parser.add_argument(
        "--visual_dim",
        type=int,
        default=128,
        help="Dimension of BiSST visual feature.",
    )
    parser.add_argument(
        "--sensor_dim",
        type=int,
        default=9,
        help="Dimension of low-dimensional robot/sensor state.",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=256,
        help="Hidden dimension of fusion MLP.",
    )
    parser.add_argument(
        "--output_dim",
        type=int,
        default=256,
        help="Output dimension of fused state feature.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for test.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device: cuda or cpu.",
    )

    return parser.parse_args()


class StateEncoder(nn.Module):
    """
    Multimodal state encoder for RL frontend.

    Inputs:
        visual_feature: [B, visual_dim]
            Feature extracted by BiSSTVisualEncoder.

        sensor_state: [B, sensor_dim]
            Low-dimensional robot/sensor state.
            For example:
                - insertion position
                - bending state
                - rotation state
                - EM pose
                - target direction
                - previous action

    Output:
        fused_feature: [B, output_dim]
            Multimodal feature used by LSTM / policy network.
    """

    def __init__(
        self,
        visual_dim: int = 128,
        sensor_dim: int = 14,
        hidden_dim: int = 256,
        output_dim: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.visual_dim = visual_dim
        self.sensor_dim = sensor_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout

        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.sensor_proj = nn.Sequential(
            nn.Linear(sensor_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

        fusion_layers = [
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        ]

        if dropout > 0:
            fusion_layers.append(nn.Dropout(dropout))

        fusion_layers.extend(
            [
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.ReLU(inplace=True),
            ]
        )

        self.fusion = nn.Sequential(*fusion_layers)

    def forward(
        self,
        visual_feature: torch.Tensor,
        sensor_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            visual_feature: [B, visual_dim]
            sensor_state: [B, sensor_dim]

        Returns:
            fused_feature: [B, output_dim]
        """
        if visual_feature.dim() != 2:
            raise ValueError(
                "visual_feature should have shape [B, visual_dim], got {}".format(
                    tuple(visual_feature.shape)
                )
            )

        if sensor_state.dim() != 2:
            raise ValueError(
                "sensor_state should have shape [B, sensor_dim], got {}".format(
                    tuple(sensor_state.shape)
                )
            )

        if visual_feature.shape[0] != sensor_state.shape[0]:
            raise ValueError(
                "Batch size mismatch: visual_feature batch {}, sensor_state batch {}".format(
                    visual_feature.shape[0],
                    sensor_state.shape[0],
                )
            )

        if visual_feature.shape[1] != self.visual_dim:
            raise ValueError(
                "visual_feature dim mismatch: expected {}, got {}".format(
                    self.visual_dim,
                    visual_feature.shape[1],
                )
            )

        if sensor_state.shape[1] != self.sensor_dim:
            raise ValueError(
                "sensor_state dim mismatch: expected {}, got {}".format(
                    self.sensor_dim,
                    sensor_state.shape[1],
                )
            )

        visual_emb = self.visual_proj(visual_feature)
        sensor_emb = self.sensor_proj(sensor_state)

        fused = torch.cat(
            [visual_emb, sensor_emb],
            dim=1,
        )

        fused_feature = self.fusion(fused)

        return fused_feature


def main():
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Fall back to CPU.")
        args.device = "cpu"

    encoder = StateEncoder(
        visual_dim=args.visual_dim,
        sensor_dim=args.sensor_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        dropout=0.0,
    ).to(args.device)

    encoder.eval()

    visual_feature = torch.randn(
        args.batch_size,
        args.visual_dim,
        device=args.device,
    )

    sensor_state = torch.randn(
        args.batch_size,
        args.sensor_dim,
        device=args.device,
    )

    with torch.no_grad():
        fused_feature = encoder(
            visual_feature=visual_feature,
            sensor_state=sensor_state,
        )

    print("========== State Encoder Test ==========")
    print("Visual feature shape:", visual_feature.shape)
    print("Sensor state shape:", sensor_state.shape)
    print("Fused feature shape:", fused_feature.shape)
    print("Fused feature dtype:", fused_feature.dtype)
    print("Fused feature device:", fused_feature.device)
    print("Fused feature mean:", fused_feature.mean().item())
    print("Fused feature std:", fused_feature.std().item())


if __name__ == "__main__":
    main()