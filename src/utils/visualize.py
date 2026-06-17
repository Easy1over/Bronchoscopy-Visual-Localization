from pathlib import Path
from typing import Union, List

import torch
from PIL import Image
import numpy as np


def denormalize_tensor(image_tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert image tensor from [-1, 1] to [0, 1].
    """
    image_tensor = (image_tensor + 1.0) / 2.0
    image_tensor = torch.clamp(image_tensor, 0.0, 1.0)
    return image_tensor


def tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    """
    Convert a tensor image to PIL image.

    Input:
        Tensor shape [3, H, W], range [-1, 1] or [0, 1]
    """
    if image_tensor.dim() != 3:
        raise ValueError(
            "Expected image tensor with shape [3, H, W], but got {}".format(
                tuple(image_tensor.shape)
            )
        )

    if image_tensor.size(0) != 3:
        raise ValueError(
            "Expected 3 channels, but got {}".format(image_tensor.size(0))
        )

    if image_tensor.min().item() < 0:
        image_tensor = denormalize_tensor(image_tensor)

    image_tensor = image_tensor.detach().cpu()
    image_array = image_tensor.permute(1, 2, 0).numpy()
    image_array = (image_array * 255.0).round().astype(np.uint8)

    return Image.fromarray(image_array, mode="RGB")


def save_tensor_image(
    image_tensor: torch.Tensor,
    save_path: Union[str, Path],
) -> None:
    """
    Save a tensor image to disk.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    image = tensor_to_pil(image_tensor)
    image.save(save_path)


def make_image_grid(
    image_tensors: List[torch.Tensor],
    padding: int = 5,
) -> Image.Image:
    """
    Make a horizontal image grid from a list of tensor images.
    """
    if len(image_tensors) == 0:
        raise ValueError("image_tensors should not be empty.")

    pil_images = [tensor_to_pil(img) for img in image_tensors]

    widths, heights = zip(*(img.size for img in pil_images))

    total_width = sum(widths) + padding * (len(pil_images) - 1)
    max_height = max(heights)

    grid = Image.new("RGB", (total_width, max_height), color=(255, 255, 255))

    x_offset = 0
    for img in pil_images:
        grid.paste(img, (x_offset, 0))
        x_offset += img.size[0] + padding

    return grid


def save_image_grid(
    image_tensors: List[torch.Tensor],
    save_path: Union[str, Path],
    padding: int = 5,
) -> None:
    """
    Save a horizontal image grid.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    grid = make_image_grid(image_tensors, padding=padding)
    grid.save(save_path)