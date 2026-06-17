# src/datasets/mask_dataset.py

import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
)


def list_image_files(
    image_dir: str,
) -> List[Path]:
    """
    List image files under a directory.

    Args:
        image_dir:
            Directory containing images.

    Returns:
        Sorted list of image paths.
    """
    image_dir_path = Path(image_dir)

    if not image_dir_path.exists():
        raise FileNotFoundError(
            "Image directory does not exist: {}".format(image_dir_path)
        )

    image_paths = []

    for path in image_dir_path.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            image_paths.append(path)

    image_paths = sorted(image_paths)

    return image_paths


class PairedMaskDataset(Dataset):
    """
    Paired image-mask dataset for training Fmask.

    Expected structure example:
        image_dir/
            rgb0001.png
            rgb0002.png

        mask_dir/
            mask0001.png
            mask0002.png

    Also supports:
        image_dir/
            0001.png

        mask_dir/
            0001.png

    Matching rule:
        normalize_pair_stem(image_stem) == normalize_pair_stem(mask_stem)

    Examples:
        rgb0001  -> 0001
        mask0001 -> 0001
        0001     -> 0001

    Returned item:
        {
            "image": image tensor, shape [3, H, W], range [-1, 1]
            "mask": mask tensor, shape [1, H, W], range [0, 1]
            "image_path": str
            "mask_path": str
        }

    This dataset is only for training the lightweight mask predictor Fmask.
    It is separate from the unpaired BiSST image translation dataset.
    """

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        image_size: int = 256,
        train: bool = True,
        binarize_threshold: float = 0.5,
    ) -> None:
        super().__init__()

        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = image_size
        self.train = train
        self.binarize_threshold = binarize_threshold

        self.image_paths = list_image_files(str(self.image_dir))
        self.mask_paths = list_image_files(str(self.mask_dir))

        if len(self.image_paths) == 0:
            raise RuntimeError(
                "No image files found in image_dir: {}".format(self.image_dir)
            )

        if len(self.mask_paths) == 0:
            raise RuntimeError(
                "No mask files found in mask_dir: {}".format(self.mask_dir)
            )

        self.mask_map = self.build_mask_map(self.mask_paths)
        self.pairs = self.build_pairs(
            image_paths=self.image_paths,
            mask_map=self.mask_map,
        )

        if len(self.pairs) == 0:
            raise RuntimeError(
                "No paired image-mask files found. "
                "Please check whether image and mask stems match. "
                "Current matching supports examples like rgb0001 <-> mask0001."
            )

        print(
            "PairedMaskDataset: found {} images, {} masks, {} pairs.".format(
                len(self.image_paths),
                len(self.mask_paths),
                len(self.pairs),
            )
        )

    @staticmethod
    def normalize_pair_stem(
        stem: str,
    ) -> str:
        """
        Normalize image / mask filename stem for pairing.

        Examples:
            rgb0001       -> 0001
            mask0001      -> 0001
            image0001     -> 0001
            img0001       -> 0001
            0001          -> 0001

        This function is intentionally simple and conservative.
        """
        stem = stem.lower()

        prefixes = [
            "rgb",
            "mask",
            "image",
            "img",
        ]

        for prefix in prefixes:
            if stem.startswith(prefix):
                stem = stem[len(prefix):]
                break

        # Remove common separators after prefix, e.g. rgb_0001 -> 0001
        while stem.startswith("_") or stem.startswith("-") or stem.startswith(" "):
            stem = stem[1:]

        return stem

    @staticmethod
    def build_mask_map(
        mask_paths: List[Path],
    ) -> Dict[str, Path]:
        """
        Build normalized stem -> mask path map.

        Example:
            mask0001.png -> key 0001
        """
        mask_map = {}

        for path in mask_paths:
            key = PairedMaskDataset.normalize_pair_stem(path.stem)

            if key in mask_map:
                print(
                    "Warning: duplicate normalized mask key '{}'. "
                    "Using later file: {}".format(key, path)
                )

            mask_map[key] = path

        return mask_map

    @staticmethod
    def build_pairs(
        image_paths: List[Path],
        mask_map: Dict[str, Path],
    ) -> List[Tuple[Path, Path]]:
        """
        Match image paths and mask paths by normalized file stem.

        Example:
            rgb0001.png  <->  mask0001.png
        """
        pairs = []

        for image_path in image_paths:
            key = PairedMaskDataset.normalize_pair_stem(image_path.stem)

            if key in mask_map:
                pairs.append(
                    (
                        image_path,
                        mask_map[key],
                    )
                )

        return pairs

    def __len__(
        self,
    ) -> int:
        return len(self.pairs)

    def load_image(
        self,
        path: Path,
    ) -> Image.Image:
        """
        Load RGB image.
        """
        image = Image.open(path).convert("RGB")
        return image

    def load_mask(
        self,
        path: Path,
    ) -> Image.Image:
        """
        Load mask image as grayscale.
        """
        mask = Image.open(path).convert("L")
        return mask

    def resize_image(
        self,
        image: Image.Image,
    ) -> Image.Image:
        """
        Resize RGB image.
        """
        resize = transforms.Resize(
            (self.image_size, self.image_size),
            interpolation=transforms.InterpolationMode.BICUBIC,
        )

        return resize(image)

    def resize_mask(
        self,
        mask: Image.Image,
    ) -> Image.Image:
        """
        Resize mask with nearest interpolation.
        """
        resize = transforms.Resize(
            (self.image_size, self.image_size),
            interpolation=transforms.InterpolationMode.NEAREST,
        )

        return resize(mask)

    def image_to_tensor(
        self,
        image: Image.Image,
    ) -> torch.Tensor:
        """
        Convert RGB image to tensor and normalize to [-1, 1].
        """
        image_tensor = transforms.ToTensor()(image)

        image_tensor = transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        )(image_tensor)

        return image_tensor

    def mask_to_tensor(
        self,
        mask: Image.Image,
    ) -> torch.Tensor:
        """
        Convert mask to binary tensor in [0, 1].
        """
        mask_tensor = transforms.ToTensor()(mask)
        mask_tensor = (mask_tensor >= self.binarize_threshold).float()

        return mask_tensor

    def __getitem__(
        self,
        index: int,
    ) -> Dict[str, Any]:
        image_path, mask_path = self.pairs[index]

        image = self.load_image(image_path)
        mask = self.load_mask(mask_path)

        image = self.resize_image(image)
        mask = self.resize_mask(mask)

        # Important:
        # Use the same random flip for image and mask.
        if self.train:
            import random

            do_flip = random.random() < 0.5

            if do_flip:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

        image_tensor = self.image_to_tensor(image)
        mask_tensor = self.mask_to_tensor(mask)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
        }


def build_mask_dataset(
    image_dir: str,
    mask_dir: str,
    image_size: int = 256,
    train: bool = True,
    binarize_threshold: float = 0.5,
) -> PairedMaskDataset:
    """
    Build paired mask dataset.
    """
    dataset = PairedMaskDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        image_size=image_size,
        train=train,
        binarize_threshold=binarize_threshold,
    )

    return dataset


if __name__ == "__main__":
    print("PairedMaskDataset self-test")

    if len(sys.argv) < 3:
        print("")
        print("Usage:")
        print("  python src/datasets/mask_dataset.py IMAGE_DIR MASK_DIR")
        print("")
        print("Example:")
        print("  python src/datasets/mask_dataset.py data/masks/images data/masks/masks")
        sys.exit(0)

    image_dir_arg = sys.argv[1]
    mask_dir_arg = sys.argv[2]

    dataset = build_mask_dataset(
        image_dir=image_dir_arg,
        mask_dir=mask_dir_arg,
        image_size=256,
        train=True,
        binarize_threshold=0.5,
    )

    print("Dataset size:", len(dataset))

    sample = dataset[0]

    print("image shape:", sample["image"].shape)
    print("mask shape:", sample["mask"].shape)
    print("image min/max:", sample["image"].min().item(), sample["image"].max().item())
    print("mask min/max:", sample["mask"].min().item(), sample["mask"].max().item())
    print("image path:", sample["image_path"])
    print("mask path:", sample["mask_path"])

    print("")
    print("First 5 pairs:")
    for i in range(min(5, len(dataset.pairs))):
        image_path, mask_path = dataset.pairs[i]
        print("{}  <->  {}".format(image_path.name, mask_path.name))