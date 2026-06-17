# src/datasets/unpaired_dataset.py

import random
from pathlib import Path
from typing import Dict, Any, List

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


def list_image_files(image_dir: str) -> List[Path]:
    """
    List image files in a directory.

    Supported formats:
        jpg, jpeg, png, bmp, tif, tiff
    """
    image_dir = Path(image_dir)

    if not image_dir.exists():
        raise FileNotFoundError(
            "Image directory does not exist: {}".format(image_dir)
        )

    suffixes = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
    }

    image_paths = [
        path for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    ]

    image_paths = sorted(image_paths)

    return image_paths


def build_image_transform(
    image_size: int,
    train: bool = True,
):
    """
    Build image transform.

    Output range:
        [-1, 1]
    """
    transform_list = []

    transform_list.append(
        transforms.Resize(
            (image_size, image_size),
            interpolation=transforms.InterpolationMode.BICUBIC,
        )
    )

    if train:
        transform_list.append(
            transforms.RandomHorizontalFlip(p=0.5)
        )

    transform_list.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
            ),
        ]
    )

    return transforms.Compose(transform_list)


class UnpairedBronchoscopyDataset(Dataset):
    """
    Unpaired bronchoscopy dataset.

    Direction:
        virtual -> real

    Output:
        {
            "virtual": Tensor [3, H, W],
            "real": Tensor [3, H, W],
            "virtual_path": str,
            "real_path": str,
        }

    Notes:
        - virtual and real images are unpaired.
        - during training, real image is randomly sampled.
        - during testing, real image is selected deterministically.
    """

    def __init__(
        self,
        virtual_dir: str,
        real_dir: str,
        image_size: int = 256,
        train: bool = True,
    ) -> None:
        super().__init__()

        self.virtual_dir = Path(virtual_dir)
        self.real_dir = Path(real_dir)

        self.image_size = image_size
        self.train = train

        self.virtual_paths = list_image_files(self.virtual_dir)
        self.real_paths = list_image_files(self.real_dir)

        if len(self.virtual_paths) == 0:
            raise RuntimeError(
                "No virtual images found in {}".format(self.virtual_dir)
            )

        if len(self.real_paths) == 0:
            raise RuntimeError(
                "No real images found in {}".format(self.real_dir)
            )

        self.image_transform = build_image_transform(
            image_size=image_size,
            train=train,
        )

    def load_rgb_image(
        self,
        image_path: Path,
    ) -> Image.Image:
        image = Image.open(image_path).convert("RGB")
        return image

    def __len__(self) -> int:
        return len(self.virtual_paths)

    def __getitem__(
        self,
        index: int,
    ) -> Dict[str, Any]:
        virtual_path = self.virtual_paths[index % len(self.virtual_paths)]

        if self.train:
            real_index = random.randint(0, len(self.real_paths) - 1)
        else:
            real_index = index % len(self.real_paths)

        real_path = self.real_paths[real_index]

        virtual_image = self.load_rgb_image(virtual_path)
        real_image = self.load_rgb_image(real_path)

        virtual_tensor = self.image_transform(virtual_image)
        real_tensor = self.image_transform(real_image)

        sample = {
            "virtual": virtual_tensor,
            "real": real_tensor,
            "virtual_path": str(virtual_path),
            "real_path": str(real_path),
        }

        return sample


if __name__ == "__main__":
    dataset = UnpairedBronchoscopyDataset(
        virtual_dir="data/processed/virtual",
        real_dir="data/processed/real",
        image_size=128,
        train=True,
    )

    sample = dataset[0]

    print("Dataset test")
    print("Dataset size:", len(dataset))

    for key, value in sample.items():
        if torch.is_tensor(value):
            print("{} shape: {}".format(key, value.shape))
        else:
            print("{}: {}".format(key, value))