from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image


def denormalize(image_tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert image tensor from [-1, 1] to [0, 1].

    Args:
        image_tensor: Tensor with shape [3, H, W] or [B, 3, H, W]

    Returns:
        Tensor in [0, 1]
    """
    return torch.clamp((image_tensor + 1.0) / 2.0, 0.0, 1.0)


def tensor_to_numpy(image_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert image tensor to uint8 numpy image.

    Args:
        image_tensor: Tensor with shape [3, H, W], range [-1, 1] or [0, 1]

    Returns:
        numpy array with shape [H, W, 3], dtype uint8
    """
    if image_tensor.dim() != 3:
        raise ValueError(
            "Expected tensor shape [3, H, W], got {}".format(
                tuple(image_tensor.shape)
            )
        )

    if image_tensor.size(0) != 3:
        raise ValueError(
            "Expected 3 channels, got {}".format(image_tensor.size(0))
        )

    image_tensor = image_tensor.detach().cpu()

    if image_tensor.min().item() < 0:
        image_tensor = denormalize(image_tensor)

    image_tensor = torch.clamp(image_tensor, 0.0, 1.0)
    image_numpy = image_tensor.permute(1, 2, 0).numpy()
    image_numpy = (image_numpy * 255.0).round().astype(np.uint8)

    return image_numpy


def tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    """
    Convert image tensor to PIL RGB image.
    """
    image_numpy = tensor_to_numpy(image_tensor)
    return Image.fromarray(image_numpy)


def save_tensor_image(
    image_tensor: torch.Tensor,
    save_path: Union[str, Path],
) -> None:
    """
    Save image tensor to disk.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    image = tensor_to_pil(image_tensor)
    image.save(save_path)


def save_image_batch(
    image_batch: torch.Tensor,
    save_dir: Union[str, Path],
    prefix: str = "image",
) -> None:
    """
    Save a batch of image tensors.

    Args:
        image_batch: Tensor with shape [B, 3, H, W]
        save_dir: output directory
        prefix: filename prefix
    """
    if image_batch.dim() != 4:
        raise ValueError(
            "Expected batch tensor shape [B, 3, H, W], got {}".format(
                tuple(image_batch.shape)
            )
        )

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    batch_size = image_batch.size(0)

    for i in range(batch_size):
        save_path = save_dir / "{}_{:04d}.png".format(prefix, i)
        save_tensor_image(image_batch[i], save_path)


if __name__ == "__main__":
    x = torch.rand(3, 128, 128) * 2.0 - 1.0

    out_path = Path("results") / "debug" / "image_utils_test.png"
    save_tensor_image(x, out_path)

    print("Image utils test")
    print("Input shape:", x.shape)
    print("Saved to:", out_path)