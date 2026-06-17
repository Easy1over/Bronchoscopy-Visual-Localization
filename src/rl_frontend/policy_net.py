# src/rl_frontend/policy_net.py

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.distributions as distributions

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.rl_frontend.state_encoder import StateEncoder


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test recurrent actor-critic policy network for RL frontend."
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
        default=14,
        help="Dimension of low-dimensional robot/sensor state.",
    )
    parser.add_argument(
        "--fusion_hidden_dim",
        type=int,
        default=256,
        help="Hidden dimension of StateEncoder fusion MLP.",
    )
    parser.add_argument(
        "--fusion_output_dim",
        type=int,
        default=256,
        help="Output dimension of StateEncoder.",
    )
    parser.add_argument(
        "--lstm_hidden_dim",
        type=int,
        default=512,
        help="Hidden dimension of LSTM. Paper-style default is 512.",
    )
    parser.add_argument(
        "--lstm_layers",
        type=int,
        default=1,
        help="Number of LSTM layers.",
    )
    parser.add_argument(
        "--action_dim",
        type=int,
        default=3,
        help="Action dimension. For bronchoscope control: [d, rx, ry].",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for test.",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=8,
        help="Sequence length for test.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device: cuda or cpu.",
    )

    return parser.parse_args()


class RecurrentActorCriticPolicy(nn.Module):
    """
    Recurrent actor-critic policy for bronchoscopy navigation.

    Inputs:
        visual_feature_seq: [B, T, visual_dim]
            BiSST visual structural feature sequence.

        sensor_state_seq: [B, T, sensor_dim]
            Low-dimensional robot/sensor state sequence.

    Outputs:
        action_mean: [B, T, action_dim]
        action_std: [B, T, action_dim]
        value: [B, T, 1]

    Action definition:
        action = [d, rx, ry]
            d: insertion command
            rx: bending / joystick x command
            ry: rotation or joystick y command

    Notes:
        - This network can be used for BC and PPO.
        - For BC, usually use action_mean and MSE loss against expert action.
        - For PPO, sample from Normal(action_mean, action_std).
    """

    def __init__(
        self,
        visual_dim: int = 128,
        sensor_dim: int = 14,
        fusion_hidden_dim: int = 256,
        fusion_output_dim: int = 256,
        lstm_hidden_dim: int = 512,
        lstm_layers: int = 1,
        action_dim: int = 3,
        action_std_init: float = 0.5,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if lstm_layers < 1:
            raise ValueError("lstm_layers must be >= 1.")

        if action_std_init <= 0:
            raise ValueError("action_std_init must be > 0.")

        self.visual_dim = visual_dim
        self.sensor_dim = sensor_dim
        self.fusion_hidden_dim = fusion_hidden_dim
        self.fusion_output_dim = fusion_output_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layers = lstm_layers
        self.action_dim = action_dim

        self.state_encoder = StateEncoder(
            visual_dim=visual_dim,
            sensor_dim=sensor_dim,
            hidden_dim=fusion_hidden_dim,
            output_dim=fusion_output_dim,
            dropout=dropout,
        )

        self.lstm = nn.LSTM(
            input_size=fusion_output_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
        )

        self.actor = nn.Sequential(
            nn.Linear(lstm_hidden_dim, lstm_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(lstm_hidden_dim, action_dim),
            nn.Tanh(),
        )

        self.critic = nn.Sequential(
            nn.Linear(lstm_hidden_dim, lstm_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(lstm_hidden_dim, 1),
        )

        log_std_value = torch.log(torch.tensor(action_std_init))
        self.log_std = nn.Parameter(
            torch.ones(action_dim) * log_std_value
        )

    def init_hidden(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize LSTM hidden state.

        Returns:
            h0: [lstm_layers, B, lstm_hidden_dim]
            c0: [lstm_layers, B, lstm_hidden_dim]
        """
        h0 = torch.zeros(
            self.lstm_layers,
            batch_size,
            self.lstm_hidden_dim,
            device=device,
        )
        c0 = torch.zeros(
            self.lstm_layers,
            batch_size,
            self.lstm_hidden_dim,
            device=device,
        )

        return h0, c0

    def encode_sequence(
        self,
        visual_feature_seq: torch.Tensor,
        sensor_state_seq: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode multimodal sequence with StateEncoder + LSTM.

        Args:
            visual_feature_seq: [B, T, visual_dim]
            sensor_state_seq: [B, T, sensor_dim]
            hidden_state: optional LSTM hidden state

        Returns:
            lstm_output: [B, T, lstm_hidden_dim]
            next_hidden_state: tuple(h, c)
        """
        if visual_feature_seq.dim() != 3:
            raise ValueError(
                "visual_feature_seq should be [B, T, visual_dim], got {}".format(
                    tuple(visual_feature_seq.shape)
                )
            )

        if sensor_state_seq.dim() != 3:
            raise ValueError(
                "sensor_state_seq should be [B, T, sensor_dim], got {}".format(
                    tuple(sensor_state_seq.shape)
                )
            )

        if visual_feature_seq.shape[0] != sensor_state_seq.shape[0]:
            raise ValueError("Batch size mismatch between visual and sensor sequence.")

        if visual_feature_seq.shape[1] != sensor_state_seq.shape[1]:
            raise ValueError("Sequence length mismatch between visual and sensor sequence.")

        batch_size = visual_feature_seq.shape[0]
        seq_len = visual_feature_seq.shape[1]

        visual_flat = visual_feature_seq.reshape(
            batch_size * seq_len,
            self.visual_dim,
        )
        sensor_flat = sensor_state_seq.reshape(
            batch_size * seq_len,
            self.sensor_dim,
        )

        fused_flat = self.state_encoder(
            visual_feature=visual_flat,
            sensor_state=sensor_flat,
        )

        fused_seq = fused_flat.reshape(
            batch_size,
            seq_len,
            self.fusion_output_dim,
        )

        if hidden_state is None:
            hidden_state = self.init_hidden(
                batch_size=batch_size,
                device=fused_seq.device,
            )

        lstm_output, next_hidden_state = self.lstm(
            fused_seq,
            hidden_state,
        )

        return lstm_output, next_hidden_state

    def forward(
        self,
        visual_feature_seq: torch.Tensor,
        sensor_state_seq: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        """
        Forward sequence.

        Returns:
            action_mean: [B, T, action_dim]
            action_std: [B, T, action_dim]
            value: [B, T, 1]
            next_hidden_state: tuple(h, c)
        """
        lstm_output, next_hidden_state = self.encode_sequence(
            visual_feature_seq=visual_feature_seq,
            sensor_state_seq=sensor_state_seq,
            hidden_state=hidden_state,
        )

        action_mean = self.actor(lstm_output)
        value = self.critic(lstm_output)

        action_std = torch.exp(self.log_std)
        action_std = action_std.view(1, 1, self.action_dim)
        action_std = action_std.expand_as(action_mean)

        return action_mean, action_std, value, next_hidden_state

    def get_action_distribution(
        self,
        visual_feature_seq: torch.Tensor,
        sensor_state_seq: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        """
        Build Normal action distribution for PPO.

        Returns:
            dist: torch.distributions.Normal
            value: [B, T, 1]
            next_hidden_state: tuple(h, c)
        """
        action_mean, action_std, value, next_hidden_state = self.forward(
            visual_feature_seq=visual_feature_seq,
            sensor_state_seq=sensor_state_seq,
            hidden_state=hidden_state,
        )

        dist = distributions.Normal(
            loc=action_mean,
            scale=action_std,
        )

        return dist, value, next_hidden_state

    def act(
        self,
        visual_feature: torch.Tensor,
        sensor_state: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        deterministic: bool = False,
    ):
        """
        Single-step action.

        Args:
            visual_feature: [B, visual_dim]
            sensor_state: [B, sensor_dim]
            hidden_state: optional LSTM hidden state
            deterministic:
                True: use action mean
                False: sample from Normal distribution

        Returns:
            action: [B, action_dim]
            log_prob: [B]
            value: [B, 1]
            next_hidden_state: tuple(h, c)
        """
        if visual_feature.dim() != 2:
            raise ValueError(
                "visual_feature should be [B, visual_dim], got {}".format(
                    tuple(visual_feature.shape)
                )
            )

        if sensor_state.dim() != 2:
            raise ValueError(
                "sensor_state should be [B, sensor_dim], got {}".format(
                    tuple(sensor_state.shape)
                )
            )

        visual_seq = visual_feature.unsqueeze(1)
        sensor_seq = sensor_state.unsqueeze(1)

        dist, value_seq, next_hidden_state = self.get_action_distribution(
            visual_feature_seq=visual_seq,
            sensor_state_seq=sensor_seq,
            hidden_state=hidden_state,
        )

        if deterministic:
            action_seq = dist.mean
        else:
            action_seq = dist.rsample()

        log_prob_seq = dist.log_prob(action_seq).sum(dim=-1)

        action = action_seq[:, 0, :]
        log_prob = log_prob_seq[:, 0]
        value = value_seq[:, 0, :]

        return action, log_prob, value, next_hidden_state

    def evaluate_actions(
        self,
        visual_feature_seq: torch.Tensor,
        sensor_state_seq: torch.Tensor,
        action_seq: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        """
        Evaluate given actions for PPO.

        Args:
            visual_feature_seq: [B, T, visual_dim]
            sensor_state_seq: [B, T, sensor_dim]
            action_seq: [B, T, action_dim]

        Returns:
            log_prob: [B, T]
            entropy: [B, T]
            value: [B, T, 1]
            next_hidden_state: tuple(h, c)
        """
        dist, value, next_hidden_state = self.get_action_distribution(
            visual_feature_seq=visual_feature_seq,
            sensor_state_seq=sensor_state_seq,
            hidden_state=hidden_state,
        )

        log_prob = dist.log_prob(action_seq).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return log_prob, entropy, value, next_hidden_state


def main():
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Fall back to CPU.")
        args.device = "cpu"

    device = torch.device(args.device)

    policy = RecurrentActorCriticPolicy(
        visual_dim=args.visual_dim,
        sensor_dim=args.sensor_dim,
        fusion_hidden_dim=args.fusion_hidden_dim,
        fusion_output_dim=args.fusion_output_dim,
        lstm_hidden_dim=args.lstm_hidden_dim,
        lstm_layers=args.lstm_layers,
        action_dim=args.action_dim,
        action_std_init=0.5,
        dropout=0.0,
    ).to(device)

    policy.eval()

    visual_feature_seq = torch.randn(
        args.batch_size,
        args.seq_len,
        args.visual_dim,
        device=device,
    )

    sensor_state_seq = torch.randn(
        args.batch_size,
        args.seq_len,
        args.sensor_dim,
        device=device,
    )

    with torch.no_grad():
        action_mean, action_std, value, next_hidden_state = policy(
            visual_feature_seq=visual_feature_seq,
            sensor_state_seq=sensor_state_seq,
        )

        action, log_prob, single_value, single_hidden = policy.act(
            visual_feature=visual_feature_seq[:, 0, :],
            sensor_state=sensor_state_seq[:, 0, :],
            deterministic=False,
        )

        log_prob_eval, entropy, value_eval, _ = policy.evaluate_actions(
            visual_feature_seq=visual_feature_seq,
            sensor_state_seq=sensor_state_seq,
            action_seq=action_mean,
        )

    print("========== Recurrent Actor-Critic Policy Test ==========")
    print("Visual feature seq shape:", visual_feature_seq.shape)
    print("Sensor state seq shape:", sensor_state_seq.shape)
    print("Action mean shape:", action_mean.shape)
    print("Action std shape:", action_std.shape)
    print("Value shape:", value.shape)
    print("Next hidden h shape:", next_hidden_state[0].shape)
    print("Next hidden c shape:", next_hidden_state[1].shape)
    print("")
    print("Single-step action shape:", action.shape)
    print("Single-step log_prob shape:", log_prob.shape)
    print("Single-step value shape:", single_value.shape)
    print("")
    print("Evaluate log_prob shape:", log_prob_eval.shape)
    print("Evaluate entropy shape:", entropy.shape)
    print("Evaluate value shape:", value_eval.shape)
    print("")
    print("Action mean range: [{:.4f}, {:.4f}]".format(
        action_mean.min().item(),
        action_mean.max().item(),
    ))
    print("Action sample range: [{:.4f}, {:.4f}]".format(
        action.min().item(),
        action.max().item(),
    ))


if __name__ == "__main__":
    main()