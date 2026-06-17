# src/infer/infer_bisst.py

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.models.bisst import BiSSTModel
from src.utils.visualize import save_image_grid
from src.utils.experiment import create_run_name


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
        description="Batch inference for paper-aligned single-direction BiSST model."
    )

    # Input / output
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help=(
            "Directory of input images. If not set, it will be selected by direction: "
            "virtual2real -> data/processed/virtual, real2virtual -> data/processed/real."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained BiSST checkpoint.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=str(PROJECT_ROOT / "results" / "bisst_infer"),
        help="Root directory for inference results.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Inference run name. If not set, a timestamp-based name will be created.",
    )
    parser.add_argument(
        "--direction",
        type=str,
        default="auto",
        choices=["auto", "virtual2real", "real2virtual"],
        help=(
            "Inference direction. Use auto to read from checkpoint when possible. "
            "Old checkpoints without direction metadata will fall back to virtual2real."
        ),
    )

    # Image
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Input image size used for inference.",
    )
    parser.add_argument(
        "--keep_size",
        action="store_true",
        help="Resize generated result back to original image size before saving.",
    )

    # Batch inference
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference. Larger values can improve GPU utilization.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of dataloader workers. Use 0 first on Windows.",
    )
    parser.add_argument(
        "--print_freq",
        type=int,
        default=20,
        help="Print progress every N images.",
    )

    # Model architecture, should match training
    parser.add_argument(
        "--ngf",
        type=int,
        default=64,
        help="Number of generator filters. Must match training.",
    )
    parser.add_argument(
        "--ndf",
        type=int,
        default=64,
        help="Number of discriminator filters. Must match training.",
    )
    parser.add_argument(
        "--n_blocks",
        type=int,
        default=9,
        help="Number of ResNet blocks in generator. Must match training.",
    )

    # Dummy training-related args needed by BiSSTModel constructor
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--lambda_gan", type=float, default=1.0)
    parser.add_argument("--lambda_nce", type=float, default=1.0)
    parser.add_argument("--lambda_idt", type=float, default=0.5)
    parser.add_argument("--nce_temperature", type=float, default=0.07)
    parser.add_argument("--num_patches", type=int, default=256)

    # Mask predictor Fmask
    mask_predictor_group = parser.add_mutually_exclusive_group()
    mask_predictor_group.add_argument(
        "--use_mask_predictor",
        dest="use_mask_predictor",
        action="store_true",
        help="Construct BiSST with mask predictor Fmask.",
    )
    mask_predictor_group.add_argument(
        "--no_mask_predictor",
        dest="use_mask_predictor",
        action="store_false",
        help="Construct BiSST without mask predictor Fmask.",
    )
    parser.set_defaults(use_mask_predictor=False)

    parser.add_argument("--mask_predictor_type", type=str, default="light_unet")
    parser.add_argument("--mask_base_channels", type=int, default=32)
    parser.add_argument("--mask_checkpoint", type=str, default=None)

    freeze_mask_group = parser.add_mutually_exclusive_group()
    freeze_mask_group.add_argument(
        "--freeze_mask_predictor",
        dest="freeze_mask_predictor",
        action="store_true",
    )
    freeze_mask_group.add_argument(
        "--train_mask_predictor",
        dest="freeze_mask_predictor",
        action="store_false",
    )
    parser.set_defaults(freeze_mask_predictor=True)

    parser.add_argument("--lambda_mask", type=float, default=0.0)
    parser.add_argument("--mask_loss_type", type=str, default="dice_consistency")

    # Embedding extractor Eh
    embedding_group = parser.add_mutually_exclusive_group()
    embedding_group.add_argument(
        "--use_embedding_extractor",
        dest="use_embedding_extractor",
        action="store_true",
        help="Construct BiSST with embedding extractor Eh.",
    )
    embedding_group.add_argument(
        "--no_embedding_extractor",
        dest="use_embedding_extractor",
        action="store_false",
        help="Construct BiSST without embedding extractor Eh.",
    )
    parser.set_defaults(use_embedding_extractor=True)

    parser.add_argument(
        "--embedding_extractor_type",
        type=str,
        default="conv",
        choices=["conv", "pool"],
    )
    parser.add_argument("--embedding_input_channels", type=int, default=None)
    parser.add_argument("--embedding_hidden_channels", type=int, default=128)
    parser.add_argument("--embedding_dim", type=int, default=128)

    parser.add_argument("--lambda_semantic", type=float, default=0.0)
    parser.add_argument("--lambda_semantic_neg", type=float, default=0.1)
    parser.add_argument("--semantic_loss_type", type=str, default="structural")

    detach_semantic_weight_group = parser.add_mutually_exclusive_group()
    detach_semantic_weight_group.add_argument(
        "--detach_semantic_negative_weight",
        dest="detach_semantic_negative_weight",
        action="store_true",
    )
    detach_semantic_weight_group.add_argument(
        "--no_detach_semantic_negative_weight",
        dest="detach_semantic_negative_weight",
        action="store_false",
    )
    parser.set_defaults(detach_semantic_negative_weight=True)

    # Saving
    parser.add_argument(
        "--save_comparison",
        action="store_true",
        help="Save input | fake comparison images.",
    )
    parser.add_argument(
        "--save_input",
        action="store_true",
        help="Also save normalized input images.",
    )
    parser.add_argument(
        "--output_ext",
        type=str,
        default="png",
        choices=["png", "jpg", "jpeg"],
        help="Output image extension. jpg/jpeg is usually faster and smaller than png.",
    )
    parser.add_argument(
        "--jpg_quality",
        type=int,
        default=95,
        help="JPEG quality when output_ext is jpg/jpeg.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Use strict=True when loading generator weights.",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device: cuda or cpu.",
    )

    return parser.parse_args()


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def collect_image_paths(input_dir: str) -> List[Path]:
    root = Path(input_dir)

    if not root.exists():
        raise FileNotFoundError("Input directory does not exist: {}".format(root))

    image_paths = []

    for path in root.rglob("*"):
        if path.is_file() and is_image_file(path):
            image_paths.append(path)

    image_paths = sorted(image_paths)

    return image_paths


def safe_stem_from_relative_path(
    image_path: Path,
    input_root: Path,
) -> str:
    rel_path = image_path.relative_to(input_root)
    parts = list(rel_path.parts)

    if len(parts) == 0:
        return image_path.stem

    parts[-1] = Path(parts[-1]).stem

    return "_".join(parts)


def image_to_tensor(
    image: Image.Image,
    image_size: int,
) -> torch.Tensor:
    """
    Convert PIL RGB image to tensor in [-1, 1], shape [3, H, W].
    """
    image = image.convert("RGB")
    image = image.resize((image_size, image_size), Image.BICUBIC)

    array = np.array(image, dtype=np.uint8)
    tensor = torch.from_numpy(array).permute(2, 0, 1).float() / 255.0
    tensor = tensor * 2.0 - 1.0

    return tensor


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convert tensor in [-1, 1] or [0, 1] to PIL RGB image.
    Accepts shape [3, H, W] or [1, 3, H, W].
    """
    if tensor.dim() == 4:
        tensor = tensor[0]

    tensor = tensor.detach().cpu()

    if tensor.min().item() < 0.0:
        tensor = (tensor + 1.0) / 2.0

    tensor = tensor.clamp(0.0, 1.0)
    tensor = tensor * 255.0
    tensor = tensor.byte()
    tensor = tensor.permute(1, 2, 0).contiguous()

    array = tensor.numpy()

    return Image.fromarray(array)


def resize_tensor_to_original(
    tensor: torch.Tensor,
    original_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Resize tensor [1, 3, H, W] or [3, H, W] to original PIL size.

    original_size is (width, height).
    """
    squeeze_back = False

    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
        squeeze_back = True

    original_width, original_height = original_size

    resized = F.interpolate(
        tensor,
        size=(original_height, original_width),
        mode="bilinear",
        align_corners=False,
    )

    if squeeze_back:
        resized = resized[0]

    return resized


def load_checkpoint_for_metadata(
    checkpoint_path: str,
):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Checkpoint does not exist: {}".format(checkpoint_path)
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    return checkpoint


def get_direction_from_checkpoint(
    checkpoint,
) -> Optional[str]:
    """
    Try to read direction from checkpoint.

    Supported possible formats:
        checkpoint["direction"]
        checkpoint["extra"]["direction"]
        checkpoint["extra"]["args"]["direction"]
        checkpoint["args"]["direction"]

    If not found, return None.
    """
    valid_directions = ["virtual2real", "real2virtual"]

    if not isinstance(checkpoint, dict):
        return None

    direction = checkpoint.get("direction", None)
    if direction in valid_directions:
        return direction

    if "extra" in checkpoint and isinstance(checkpoint["extra"], dict):
        extra = checkpoint["extra"]

        direction = extra.get("direction", None)
        if direction in valid_directions:
            return direction

        if "args" in extra and isinstance(extra["args"], dict):
            direction = extra["args"].get("direction", None)
            if direction in valid_directions:
                return direction

    if "args" in checkpoint and isinstance(checkpoint["args"], dict):
        direction = checkpoint["args"].get("direction", None)
        if direction in valid_directions:
            return direction

    return None


def resolve_direction(
    requested_direction: str,
    checkpoint,
) -> str:
    if requested_direction in ["virtual2real", "real2virtual"]:
        return requested_direction

    checkpoint_direction = get_direction_from_checkpoint(checkpoint)

    if checkpoint_direction is not None:
        return checkpoint_direction

    print(
        "Warning: direction=auto but checkpoint does not contain direction metadata. "
        "Fall back to virtual2real for compatibility with old checkpoints."
    )

    return "virtual2real"


def default_input_dir_for_direction(
    direction: str,
) -> str:
    if direction == "virtual2real":
        return str(PROJECT_ROOT / "data" / "processed" / "virtual")

    if direction == "real2virtual":
        return str(PROJECT_ROOT / "data" / "processed" / "real")

    raise ValueError("Unsupported direction: {}".format(direction))


def fake_folder_name_for_direction(
    direction: str,
) -> str:
    if direction == "virtual2real":
        return "fake_real"

    if direction == "real2virtual":
        return "fake_virtual"

    raise ValueError("Unsupported direction: {}".format(direction))


def input_folder_name_for_direction(
    direction: str,
) -> str:
    if direction == "virtual2real":
        return "input_virtual"

    if direction == "real2virtual":
        return "input_real"

    raise ValueError("Unsupported direction: {}".format(direction))


def fake_suffix_for_direction(
    direction: str,
) -> str:
    if direction == "virtual2real":
        return "fake_real"

    if direction == "real2virtual":
        return "fake_virtual"

    raise ValueError("Unsupported direction: {}".format(direction))


def input_suffix_for_direction(
    direction: str,
) -> str:
    if direction == "virtual2real":
        return "input_virtual"

    if direction == "real2virtual":
        return "input_real"

    raise ValueError("Unsupported direction: {}".format(direction))


class InferenceImageDataset(Dataset):
    def __init__(
        self,
        input_dir: str,
        image_size: int,
    ):
        self.input_root = Path(input_dir)
        self.image_size = image_size
        self.image_paths = collect_image_paths(input_dir)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        image = Image.open(image_path).convert("RGB")
        original_size = image.size

        input_tensor = image_to_tensor(
            image=image,
            image_size=self.image_size,
        )

        safe_stem = safe_stem_from_relative_path(
            image_path=image_path,
            input_root=self.input_root,
        )

        sample = {
            "input": input_tensor,
            "path": str(image_path),
            "safe_stem": safe_stem,
            "original_width": original_size[0],
            "original_height": original_size[1],
        }

        return sample


def collate_inference_batch(batch):
    input_tensors = []
    paths = []
    safe_stems = []
    original_sizes = []

    for item in batch:
        input_tensors.append(item["input"])
        paths.append(item["path"])
        safe_stems.append(item["safe_stem"])
        original_sizes.append(
            (
                int(item["original_width"]),
                int(item["original_height"]),
            )
        )

    inputs = torch.stack(input_tensors, dim=0)

    return {
        "input": inputs,
        "paths": paths,
        "safe_stems": safe_stems,
        "original_sizes": original_sizes,
    }


def build_model(args) -> BiSSTModel:
    model = BiSSTModel(
        input_nc=3,
        output_nc=3,
        ngf=args.ngf,
        ndf=args.ndf,
        n_blocks=args.n_blocks,
        lr=args.lr,
        beta1=args.beta1,
        beta2=args.beta2,
        lambda_gan=args.lambda_gan,
        lambda_nce=args.lambda_nce,
        lambda_idt=args.lambda_idt,
        nce_temperature=args.nce_temperature,
        num_patches=args.num_patches,
        use_mask_predictor=args.use_mask_predictor,
        mask_predictor_type=args.mask_predictor_type,
        mask_base_channels=args.mask_base_channels,
        mask_checkpoint=args.mask_checkpoint,
        freeze_mask_predictor=args.freeze_mask_predictor,
        lambda_mask=args.lambda_mask,
        mask_loss_type=args.mask_loss_type,
        use_embedding_extractor=args.use_embedding_extractor,
        embedding_extractor_type=args.embedding_extractor_type,
        embedding_input_channels=args.embedding_input_channels,
        embedding_hidden_channels=args.embedding_hidden_channels,
        embedding_dim=args.embedding_dim,
        lambda_semantic=args.lambda_semantic,
        lambda_semantic_neg=args.lambda_semantic_neg,
        semantic_loss_type=args.semantic_loss_type,
        detach_semantic_negative_weight=args.detach_semantic_negative_weight,
        device=args.device,
        direction=args.direction,
    )

    return model


def load_generator_from_checkpoint(
    model: BiSSTModel,
    checkpoint,
    checkpoint_path: str,
    device: str,
    strict: bool,
):
    if "models" not in checkpoint:
        raise KeyError("Checkpoint does not contain key 'models'.")

    saved_models = checkpoint["models"]
    current_models = model.get_model_dict()

    if "G" not in current_models:
        raise KeyError("Current BiSSTModel does not contain model key 'G'.")

    if "G" not in saved_models:
        available_keys = sorted(list(saved_models.keys()))
        raise KeyError(
            "Checkpoint does not contain generator key 'G'. "
            "Available model keys: {}".format(available_keys)
        )

    netG = current_models["G"]

    load_result = netG.load_state_dict(
        saved_models["G"],
        strict=strict,
    )

    netG.to(device)
    netG.eval()

    return netG, load_result


def save_pil_image(
    image: Image.Image,
    save_path: Path,
    output_ext: str,
    jpg_quality: int,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if output_ext in ["jpg", "jpeg"]:
        image.save(
            save_path,
            quality=jpg_quality,
            optimize=False,
        )
    else:
        image.save(save_path)


def save_infer_config(
    save_path: Path,
    args,
    image_count: int,
) -> None:
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("run_name: {}\n".format(args.run_name))
        f.write("project_root: {}\n".format(PROJECT_ROOT))

        f.write("\n[Direction]\n")
        f.write("direction: {}\n".format(args.direction))
        f.write("requested_direction: {}\n".format(args.requested_direction))

        f.write("\n[Input]\n")
        f.write("input_dir: {}\n".format(args.input_dir))
        f.write("checkpoint: {}\n".format(args.checkpoint))
        f.write("image_count: {}\n".format(image_count))

        f.write("\n[Output]\n")
        f.write("output_root: {}\n".format(args.output_root))
        f.write("save_comparison: {}\n".format(args.save_comparison))
        f.write("save_input: {}\n".format(args.save_input))
        f.write("output_ext: {}\n".format(args.output_ext))
        f.write("jpg_quality: {}\n".format(args.jpg_quality))

        f.write("\n[Image]\n")
        f.write("image_size: {}\n".format(args.image_size))
        f.write("keep_size: {}\n".format(args.keep_size))

        f.write("\n[Batch inference]\n")
        f.write("batch_size: {}\n".format(args.batch_size))
        f.write("num_workers: {}\n".format(args.num_workers))
        f.write("print_freq: {}\n".format(args.print_freq))

        f.write("\n[Model]\n")
        f.write("ngf: {}\n".format(args.ngf))
        f.write("ndf: {}\n".format(args.ndf))
        f.write("n_blocks: {}\n".format(args.n_blocks))

        f.write("\n[Mask predictor Fmask]\n")
        f.write("use_mask_predictor: {}\n".format(args.use_mask_predictor))
        f.write("mask_predictor_type: {}\n".format(args.mask_predictor_type))
        f.write("mask_base_channels: {}\n".format(args.mask_base_channels))
        f.write("mask_checkpoint: {}\n".format(args.mask_checkpoint))
        f.write("freeze_mask_predictor: {}\n".format(args.freeze_mask_predictor))

        f.write("\n[Embedding extractor Eh]\n")
        f.write("use_embedding_extractor: {}\n".format(args.use_embedding_extractor))
        f.write("embedding_extractor_type: {}\n".format(args.embedding_extractor_type))
        f.write("embedding_input_channels: {}\n".format(args.embedding_input_channels))
        f.write("embedding_hidden_channels: {}\n".format(args.embedding_hidden_channels))
        f.write("embedding_dim: {}\n".format(args.embedding_dim))

        f.write("\n[Device]\n")
        f.write("device: {}\n".format(args.device))
        f.write("strict: {}\n".format(args.strict))


def run_generator(
    netG,
    input_tensor: torch.Tensor,
) -> torch.Tensor:
    output = netG(input_tensor)

    if isinstance(output, tuple) or isinstance(output, list):
        output = output[0]

    return output


def save_one_result(
    input_tensor: torch.Tensor,
    fake_tensor: torch.Tensor,
    safe_stem: str,
    original_size: Tuple[int, int],
    fake_dir: Path,
    input_save_dir: Path,
    comparison_dir: Path,
    args,
) -> None:
    if args.keep_size:
        fake_save_tensor = resize_tensor_to_original(
            tensor=fake_tensor,
            original_size=original_size,
        )
        input_save_tensor = resize_tensor_to_original(
            tensor=input_tensor,
            original_size=original_size,
        )
    else:
        fake_save_tensor = fake_tensor
        input_save_tensor = input_tensor

    ext = args.output_ext
    fake_suffix = fake_suffix_for_direction(args.direction)
    input_suffix = input_suffix_for_direction(args.direction)

    fake_path = fake_dir / "{}_{}.{}".format(safe_stem, fake_suffix, ext)

    fake_image = tensor_to_pil(fake_save_tensor)
    save_pil_image(
        image=fake_image,
        save_path=fake_path,
        output_ext=args.output_ext,
        jpg_quality=args.jpg_quality,
    )

    if args.save_input:
        input_path = input_save_dir / "{}_{}.{}".format(safe_stem, input_suffix, ext)
        input_image = tensor_to_pil(input_save_tensor)
        save_pil_image(
            image=input_image,
            save_path=input_path,
            output_ext=args.output_ext,
            jpg_quality=args.jpg_quality,
        )

    if args.save_comparison:
        comparison_path = comparison_dir / "{}_comparison.png".format(safe_stem)

        save_image_grid(
            image_tensors=[
                input_save_tensor.detach().cpu(),
                fake_save_tensor.detach().cpu(),
            ],
            save_path=comparison_path,
            padding=5,
        )


def main():
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Fall back to CPU.")
        args.device = "cpu"

    if args.batch_size < 1:
        raise ValueError("batch_size must be >= 1.")

    if args.print_freq < 1:
        raise ValueError("print_freq must be >= 1.")

    if args.device == "cuda":
        torch.backends.cudnn.benchmark = True

    checkpoint = load_checkpoint_for_metadata(args.checkpoint)

    args.requested_direction = args.direction
    args.direction = resolve_direction(
        requested_direction=args.requested_direction,
        checkpoint=checkpoint,
    )

    if args.input_dir is None:
        args.input_dir = default_input_dir_for_direction(args.direction)

    args.run_name = create_run_name(
        prefix="bisst_infer_{}".format(args.direction),
        run_name=args.run_name,
    )

    output_root = Path(args.output_root)
    run_dir = output_root / args.run_name

    fake_dir = run_dir / fake_folder_name_for_direction(args.direction)
    input_save_dir = run_dir / input_folder_name_for_direction(args.direction)
    comparison_dir = run_dir / "comparison"

    fake_dir.mkdir(parents=True, exist_ok=True)

    if args.save_input:
        input_save_dir.mkdir(parents=True, exist_ok=True)

    if args.save_comparison:
        comparison_dir.mkdir(parents=True, exist_ok=True)

    dataset = InferenceImageDataset(
        input_dir=args.input_dir,
        image_size=args.image_size,
    )

    if len(dataset) == 0:
        raise RuntimeError(
            "No image files found in input_dir: {}".format(args.input_dir)
        )

    pin_memory = args.device == "cuda"

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=collate_inference_batch,
    )

    config_path = run_dir / "infer_config.txt"
    save_infer_config(
        save_path=config_path,
        args=args,
        image_count=len(dataset),
    )

    print("========== BiSST Batch Inference ==========")
    print("Project root:", PROJECT_ROOT)
    print("Run name:", args.run_name)
    print("Requested direction:", args.requested_direction)
    print("Resolved direction:", args.direction)
    print("Input dir:", args.input_dir)
    print("Checkpoint:", args.checkpoint)
    print("Output dir:", run_dir)
    print("Fake dir:", fake_dir)
    print("Image count:", len(dataset))
    print("Image size:", args.image_size)
    print("Keep original size:", args.keep_size)
    print("Batch size:", args.batch_size)
    print("Num workers:", args.num_workers)
    print("Output ext:", args.output_ext)
    print("Device:", args.device)
    print("Config saved to:", config_path)

    model = build_model(args)

    netG, load_result = load_generator_from_checkpoint(
        model=model,
        checkpoint=checkpoint,
        checkpoint_path=args.checkpoint,
        device=args.device,
        strict=args.strict,
    )

    print("")
    print("========== Checkpoint ==========")
    print("Loaded generator G from:", args.checkpoint)
    print("Checkpoint epoch:", checkpoint.get("epoch", "unknown"))
    print("Checkpoint step:", checkpoint.get("step", "unknown"))

    checkpoint_direction = get_direction_from_checkpoint(checkpoint)
    print(
        "Checkpoint direction:",
        checkpoint_direction if checkpoint_direction is not None else "unknown",
    )

    if not args.strict:
        print("Missing keys:", load_result.missing_keys)
        print("Unexpected keys:", load_result.unexpected_keys)

    print("")
    print("========== Translating ==========")

    processed_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_tensor = batch["input"]

            if args.device == "cuda":
                input_tensor = input_tensor.to(
                    args.device,
                    non_blocking=True,
                )
            else:
                input_tensor = input_tensor.to(args.device)

            fake_tensor = run_generator(
                netG=netG,
                input_tensor=input_tensor,
            )

            current_batch_size = input_tensor.shape[0]

            for i in range(current_batch_size):
                safe_stem = batch["safe_stems"][i]
                original_size = batch["original_sizes"][i]

                save_one_result(
                    input_tensor=input_tensor[i],
                    fake_tensor=fake_tensor[i],
                    safe_stem=safe_stem,
                    original_size=original_size,
                    fake_dir=fake_dir,
                    input_save_dir=input_save_dir,
                    comparison_dir=comparison_dir,
                    args=args,
                )

                processed_count += 1

                if (
                    processed_count == 1
                    or processed_count % args.print_freq == 0
                    or processed_count == len(dataset)
                ):
                    print(
                        "[{}/{}] saved {}".format(
                            processed_count,
                            len(dataset),
                            safe_stem,
                        )
                    )

    print("")
    print("========== Inference finished ==========")
    print("Run name:", args.run_name)
    print("Direction:", args.direction)
    print("Results saved to:", run_dir)
    print("Generated images saved to:", fake_dir)

    if args.save_comparison:
        print("Comparison images saved to:", comparison_dir)

    if args.save_input:
        print("Input images saved to:", input_save_dir)


if __name__ == "__main__":
    main()


# virtual -> real, auto input dir
# python src/infer/infer_bisst.py --checkpoint checkpoints/bisst/your_virtual2real_run/latest.pt --direction virtual2real --batch_size 8 --num_workers 0 --print_freq 50

# real -> virtual, auto input dir
# python src/infer/infer_bisst.py --checkpoint checkpoints/bisst/20260617_143913_bisst_real2virtual/latest.pt --direction real2virtual --batch_size 8 --num_workers 0 --print_freq 50
    
# real -> virtual, explicit input dir
# python src/infer/infer_bisst.py --checkpoint checkpoints/bisst/your_real2virtual_run/latest.pt --direction real2virtual --input_dir data/processed/real --batch_size 8 --num_workers 0 --print_freq 50