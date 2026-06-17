# src/rl_frontend/bisst_feature_extractor.py

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.models.bisst import BiSSTModel


def image_to_tensor(
    image_path: str,
    image_size: int,
    device: str,
) -> torch.Tensor:
    """
    Load image and convert it to tensor in [-1, 1], shape [1, 3, H, W].
    """
    image = Image.open(image_path).convert("RGB")
    image = image.resize((image_size, image_size), Image.BICUBIC)

    array = np.array(image, dtype=np.uint8)
    tensor = torch.from_numpy(array).permute(2, 0, 1).float() / 255.0
    tensor = tensor * 2.0 - 1.0
    tensor = tensor.unsqueeze(0)
    tensor = tensor.to(device)

    return tensor


def freeze_network(
    net: nn.Module,
) -> None:
    net.eval()
    for param in net.parameters():
        param.requires_grad = False


def get_last_feature(feats) -> torch.Tensor:
    if isinstance(feats, (list, tuple)):
        if len(feats) == 0:
            raise RuntimeError("Feature list is empty.")
        return feats[-1]

    if torch.is_tensor(feats):
        return feats

    raise TypeError("Unsupported feature type: {}.".format(type(feats)))


class BiSSTFeatureMapExtractor(nn.Module):
    """
    Paper-style frozen BiSST feature map extractor for RL frontend.

    It does NOT use BiSST Eh.

    Pipeline:
        image -> frozen G(return_features=True) -> structural feature map S

    Output:
        feature_map: [B, C, H, W]

    This feature map is then consumed by a trainable RLVisualProjector.
    """

    def __init__(
        self,
        checkpoint_path: str,
        input_nc: int = 3,
        output_nc: int = 3,
        ngf: int = 64,
        ndf: int = 64,
        n_blocks: int = 9,
        direction: str = "real2virtual",
        device: str = "cuda",
        strict: bool = False,
        use_mask_predictor: bool = False,
        mask_predictor_type: str = "light_unet",
        mask_base_channels: int = 32,
        mask_checkpoint: Optional[str] = None,
        freeze_mask_predictor: bool = True,
    ) -> None:
        super().__init__()

        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA is not available. Fall back to CPU.")
            device = "cpu"

        self.checkpoint_path = checkpoint_path
        self.device = device
        self.direction = direction
        self.strict = strict

        # We construct BiSSTModel only to rebuild G with the same architecture.
        # Eh is intentionally disabled here because RL frontend should use its
        # own trainable visual projector.
        self.model = BiSSTModel(
            input_nc=input_nc,
            output_nc=output_nc,
            ngf=ngf,
            ndf=ndf,
            n_blocks=n_blocks,
            lr=0.0002,
            beta1=0.5,
            beta2=0.999,
            lambda_gan=1.0,
            lambda_nce=1.0,
            lambda_idt=0.5,
            nce_temperature=0.07,
            num_patches=256,
            use_mask_predictor=use_mask_predictor,
            mask_predictor_type=mask_predictor_type,
            mask_base_channels=mask_base_channels,
            mask_checkpoint=mask_checkpoint,
            freeze_mask_predictor=freeze_mask_predictor,
            lambda_mask=0.0,
            mask_loss_type="dice_consistency",
            use_embedding_extractor=False,
            embedding_extractor_type="conv",
            embedding_input_channels=None,
            embedding_hidden_channels=128,
            embedding_dim=128,
            lambda_semantic=0.0,
            lambda_semantic_neg=0.1,
            semantic_loss_type="structural",
            detach_semantic_negative_weight=True,
            device=device,
            direction=direction,
        )

        checkpoint_path_obj = Path(checkpoint_path)
        if not checkpoint_path_obj.exists():
            raise FileNotFoundError(
                "Checkpoint does not exist: {}".format(checkpoint_path_obj)
            )

        checkpoint = torch.load(
            checkpoint_path_obj,
            map_location=device,
        )

        if "models" not in checkpoint:
            raise KeyError("Checkpoint does not contain key 'models'.")

        saved_models = checkpoint["models"]
        current_models = self.model.get_model_dict()

        if "G" not in saved_models:
            raise KeyError("Checkpoint does not contain models['G'].")

        self.G = current_models["G"]

        load_result_G = self.G.load_state_dict(
            saved_models["G"],
            strict=strict,
        )

        freeze_network(self.G)

        self.to(device)
        self.eval()

        print("Loaded frozen BiSST G from:", checkpoint_path)
        print("Direction:", direction)
        print("Use BiSST Eh in RL frontend: False")

        if not strict:
            print("G missing keys:", load_result_G.missing_keys)
            print("G unexpected keys:", load_result_G.unexpected_keys)

    def forward(
        self,
        image_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            image_tensor: [B, 3, H, W], normalized to [-1, 1]

        Returns:
            feature_map: [B, C, H', W']
        """
        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            _, feats = self.G(
                image_tensor,
                return_features=True,
            )

            feature_map = get_last_feature(feats)

        return feature_map


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test frozen BiSST G feature map extractor."
    )

    parser.add_argument("--checkpoint", type=str, required=True)
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
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--strict", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Fall back to CPU.")
        args.device = "cpu"

    extractor = BiSSTFeatureMapExtractor(
        checkpoint_path=args.checkpoint,
        ngf=args.ngf,
        ndf=args.ndf,
        n_blocks=args.n_blocks,
        direction=args.direction,
        device=args.device,
        strict=args.strict,
    )

    image_tensor = image_to_tensor(
        image_path=args.image,
        image_size=args.image_size,
        device=args.device,
    )

    feature_map = extractor(image_tensor)

    print("")
    print("========== BiSST Feature Map Extractor Test ==========")
    print("Image:", args.image)
    print("Image tensor shape:", image_tensor.shape)
    print("Feature map shape:", feature_map.shape)
    print("Feature map dtype:", feature_map.dtype)
    print("Feature map device:", feature_map.device)
    print("Feature map mean:", feature_map.mean().item())
    print("Feature map std:", feature_map.std().item())


if __name__ == "__main__":
    main()