# src/train/train_mask+
# .py

import argparse
import sys
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.datasets.mask_dataset import build_mask_dataset
from src.models.mask_predictor import build_mask_predictor
from src.losses.mask_loss import build_mask_loss
from src.utils.checkpoint import save_checkpoint
from src.utils.visualize import save_image_grid
from src.utils.experiment import (
    create_run_name,
    setup_experiment_dirs,
    save_config,
    init_loss_log,
    append_loss_log,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train lightweight mask predictor Fmask."
    )

    # Experiment
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Experiment run name. If not set, timestamp-based name will be created.",
    )

    # Data
    parser.add_argument(
        "--image_dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "masks" / "images"),
        help="Directory of RGB training images.",
    )
    parser.add_argument(
        "--mask_dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "masks" / "masks"),
        help="Directory of binary mask labels.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Training image size.",
    )
    parser.add_argument(
        "--binarize_threshold",
        type=float,
        default=0.5,
        help="Threshold used to binarize mask images after ToTensor.",
    )

    # Training
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of dataloader workers. Use 0 on Windows.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0002,
        help="Learning rate.",
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.5,
        help="Adam beta1.",
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=0.999,
        help="Adam beta2.",
    )

    # Model
    parser.add_argument(
        "--mask_predictor_type",
        type=str,
        default="light_unet",
        help="Mask predictor type.",
    )
    parser.add_argument(
        "--mask_base_channels",
        type=int,
        default=32,
        help="Base channels of lightweight mask predictor.",
    )

    # Loss
    parser.add_argument(
        "--loss_type",
        type=str,
        default="dice_bce",
        choices=["dice", "bce", "dice_bce"],
        help="Supervised mask loss type for training Fmask.",
    )

    # Logging and saving
    parser.add_argument(
        "--print_freq",
        type=int,
        default=20,
        help="Print losses every N iterations.",
    )
    parser.add_argument(
        "--log_freq",
        type=int,
        default=10,
        help="Write losses to CSV every N iterations.",
    )
    parser.add_argument(
        "--save_image_freq",
        type=int,
        default=100,
        help="Save visual predictions every N iterations.",
    )
    parser.add_argument(
        "--save_epoch_freq",
        type=int,
        default=5,
        help="Save epoch checkpoint every N epochs.",
    )
    parser.add_argument(
        "--checkpoint_root",
        type=str,
        default=str(PROJECT_ROOT / "checkpoints" / "mask_predictor"),
        help="Root directory for mask predictor checkpoints.",
    )
    parser.add_argument(
        "--result_root",
        type=str,
        default=str(PROJECT_ROOT / "results" / "mask_predictor"),
        help="Root directory for mask predictor results.",
    )

    # Resume
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint for resuming training.",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device: cuda or cpu.",
    )

    return parser.parse_args()


def format_loss_dict(loss_dict: Dict[str, float]) -> str:
    parts = []

    for key, value in loss_dict.items():
        parts.append("{}: {:.4f}".format(key, value))

    return " | ".join(parts)


def compute_mask_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> Dict[str, float]:
    """
    Compute simple Dice / IoU metrics for binary mask prediction.

    Args:
        logits:
            Predicted mask logits, shape [B, 1, H, W].

        target:
            Ground truth mask, shape [B, 1, H, W].

    Returns:
        Dictionary with dice and iou.
    """
    prob = torch.sigmoid(logits)
    pred = (prob >= threshold).float()
    target = target.float()

    if pred.shape != target.shape:
        target = torch.nn.functional.interpolate(
            target,
            size=pred.shape[-2:],
            mode="nearest",
        )

    batch_size = pred.size(0)

    pred_flat = pred.view(batch_size, -1)
    target_flat = target.view(batch_size, -1)

    intersection = (pred_flat * target_flat).sum(dim=1)

    pred_sum = pred_flat.sum(dim=1)
    target_sum = target_flat.sum(dim=1)

    union_for_dice = pred_sum + target_sum
    union_for_iou = pred_sum + target_sum - intersection

    dice = (2.0 * intersection + eps) / (union_for_dice + eps)
    iou = (intersection + eps) / (union_for_iou + eps)

    return {
        "dice": float(dice.mean().detach().cpu().item()),
        "iou": float(iou.mean().detach().cpu().item()),
    }


def save_run_info(
    save_path: Path,
    args,
    checkpoint_dir: Path,
    result_dir: Path,
    visual_dir: Path,
) -> None:
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("run_name: {}\n".format(args.run_name))
        f.write("project_root: {}\n".format(PROJECT_ROOT))

        f.write("\n[Data]\n")
        f.write("image_dir: {}\n".format(args.image_dir))
        f.write("mask_dir: {}\n".format(args.mask_dir))
        f.write("image_size: {}\n".format(args.image_size))
        f.write("binarize_threshold: {}\n".format(args.binarize_threshold))

        f.write("\n[Training]\n")
        f.write("epochs: {}\n".format(args.epochs))
        f.write("batch_size: {}\n".format(args.batch_size))
        f.write("num_workers: {}\n".format(args.num_workers))
        f.write("lr: {}\n".format(args.lr))
        f.write("beta1: {}\n".format(args.beta1))
        f.write("beta2: {}\n".format(args.beta2))

        f.write("\n[Model]\n")
        f.write("mask_predictor_type: {}\n".format(args.mask_predictor_type))
        f.write("mask_base_channels: {}\n".format(args.mask_base_channels))

        f.write("\n[Loss]\n")
        f.write("loss_type: {}\n".format(args.loss_type))

        f.write("\n[Paths]\n")
        f.write("device: {}\n".format(args.device))
        f.write("checkpoint_dir: {}\n".format(checkpoint_dir))
        f.write("result_dir: {}\n".format(result_dir))
        f.write("visual_dir: {}\n".format(visual_dir))
        f.write("resume: {}\n".format(args.resume))


def append_current_loss(
    loss_log_path: Path,
    loss_fieldnames,
    epoch: int,
    step: int,
    lr: float,
    loss_dict: Dict[str, float],
) -> None:
    log_row = {}

    for field in loss_fieldnames:
        log_row[field] = 0.0

    log_row["epoch"] = epoch
    log_row["step"] = step
    log_row["lr"] = lr

    for key, value in loss_dict.items():
        if key in log_row:
            log_row[key] = value

    append_loss_log(
        save_path=loss_log_path,
        row=log_row,
        fieldnames=loss_fieldnames,
    )


def make_mask_visual(
    mask_tensor: torch.Tensor,
) -> torch.Tensor:
    """
    Convert [1, H, W] mask tensor to [3, H, W] image-like tensor.

    Input mask should be in [0, 1].
    Output is also in [0, 1].
    """
    if mask_tensor.dim() != 3:
        raise ValueError(
            "Expected mask tensor with shape [1, H, W], got {}.".format(
                tuple(mask_tensor.shape)
            )
        )

    if mask_tensor.size(0) == 1:
        mask_tensor = mask_tensor.repeat(3, 1, 1)

    return mask_tensor


def save_training_visuals(
    model: torch.nn.Module,
    batch,
    save_dir: Path,
    epoch: int,
    step: int,
) -> None:
    """
    Save visual comparison:
        image | predicted_mask | target_mask
    """
    model.eval()

    with torch.no_grad():
        image = batch["image"]
        target_mask = batch["mask"]

        device = next(model.parameters()).device

        image_device = image.to(device)
        logits = model(image_device)
        pred_mask = torch.sigmoid(logits).detach().cpu()

    model.train()

    image_vis = image[0].detach().cpu()
    pred_vis = make_mask_visual(pred_mask[0])
    target_vis = make_mask_visual(target_mask[0].detach().cpu())

    save_path = save_dir / "epoch_{:03d}_step_{:07d}.png".format(
        epoch,
        step,
    )

    save_image_grid(
        image_tensors=[
            image_vis,
            pred_vis,
            target_vis,
        ],
        save_path=save_path,
        padding=5,
    )


def build_checkpoint_extra(args):
    return {
        "run_name": args.run_name,
        "args": vars(args),
    }


def save_latest_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_dir: Path,
    args,
    epoch: int,
    step: int,
) -> Path:
    latest_path = checkpoint_dir / "latest.pt"

    save_checkpoint(
        save_path=latest_path,
        models={
            "M": model,
            "mask_predictor": model,
        },
        optimizers={
            "M": optimizer,
            "mask_predictor": optimizer,
        },
        epoch=epoch,
        step=step,
        extra=build_checkpoint_extra(args),
    )

    return latest_path


def save_epoch_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_dir: Path,
    args,
    epoch: int,
    step: int,
) -> Path:
    epoch_path = checkpoint_dir / "epoch_{:03d}.pt".format(epoch)

    save_checkpoint(
        save_path=epoch_path,
        models={
            "M": model,
            "mask_predictor": model,
        },
        optimizers={
            "M": optimizer,
            "mask_predictor": optimizer,
        },
        epoch=epoch,
        step=step,
        extra=build_checkpoint_extra(args),
    )

    return epoch_path


def load_training_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    resume_path: str,
    device: str,
):
    checkpoint_path = Path(resume_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Resume checkpoint does not exist: {}".format(checkpoint_path)
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if "models" in checkpoint:
        models = checkpoint["models"]

        if "M" in models:
            model.load_state_dict(models["M"], strict=True)
        elif "mask_predictor" in models:
            model.load_state_dict(models["mask_predictor"], strict=True)
        else:
            raise KeyError(
                "No 'M' or 'mask_predictor' found in checkpoint['models']."
            )

    elif "model" in checkpoint:
        model.load_state_dict(checkpoint["model"], strict=True)

    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"], strict=True)

    else:
        model.load_state_dict(checkpoint, strict=True)

    if "optimizers" in checkpoint:
        optimizers = checkpoint["optimizers"]

        if "M" in optimizers:
            optimizer.load_state_dict(optimizers["M"])
        elif "mask_predictor" in optimizers:
            optimizer.load_state_dict(optimizers["mask_predictor"])
        else:
            print("Warning: no optimizer state for M / mask_predictor found.")

    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    global_step = int(checkpoint.get("step", 0))

    return start_epoch, global_step


def main():
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Fall back to CPU.")
        args.device = "cpu"

    args.run_name = create_run_name(
        prefix="mask_predictor",
        run_name=args.run_name,
    )

    dirs = setup_experiment_dirs(
        checkpoint_root=args.checkpoint_root,
        result_root=args.result_root,
        run_name=args.run_name,
    )

    checkpoint_dir = dirs["checkpoint_dir"]
    result_dir = dirs["result_dir"]
    visual_dir = dirs["visual_dir"]

    config_path = result_dir / "train_config.txt"
    run_info_path = result_dir / "run_info.txt"
    loss_log_path = result_dir / "loss_log.csv"

    save_config(args, config_path)

    save_run_info(
        save_path=run_info_path,
        args=args,
        checkpoint_dir=checkpoint_dir,
        result_dir=result_dir,
        visual_dir=visual_dir,
    )

    loss_fieldnames = [
        "epoch",
        "step",
        "lr",
        "loss",
        "dice",
        "iou",
    ]

    init_loss_log(
        save_path=loss_log_path,
        fieldnames=loss_fieldnames,
    )

    print("========== Experiment ==========")
    print("Run name:", args.run_name)
    print("Project root:", PROJECT_ROOT)
    print("Image dir:", args.image_dir)
    print("Mask dir:", args.mask_dir)
    print("Checkpoint dir:", checkpoint_dir)
    print("Result dir:", result_dir)
    print("Visual dir:", visual_dir)
    print("Loss log:", loss_log_path)
    print("Device:", args.device)

    dataset = build_mask_dataset(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        image_size=args.image_size,
        train=True,
        binarize_threshold=args.binarize_threshold,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty.")

    if len(dataloader) == 0:
        raise RuntimeError(
            "Dataloader has 0 batches. "
            "This may happen when dataset size is smaller than batch_size with drop_last=True."
        )

    print("")
    print("========== Data ==========")
    print("Dataset size:", len(dataset))
    print("Number of batches per epoch:", len(dataloader))

    model = build_mask_predictor(
        predictor_type=args.mask_predictor_type,
        input_nc=3,
        base_channels=args.mask_base_channels,
        output_nc=1,
    ).to(args.device)

    criterion = build_mask_loss(
        loss_type=args.loss_type,
        from_logits=True,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
    )

    start_epoch = 1
    global_step = 0

    if args.resume is not None:
        print("")
        print("========== Resume training ==========")
        print("Resume checkpoint:", args.resume)

        start_epoch, global_step = load_training_state(
            model=model,
            optimizer=optimizer,
            resume_path=args.resume,
            device=args.device,
        )

        print("Resume from epoch:", start_epoch)
        print("Resume global step:", global_step)

    model.train()

    for epoch in range(start_epoch, args.epochs + 1):
        print("")
        print("========== Epoch {}/{} ==========".format(epoch, args.epochs))

        last_loss_dict = None

        for batch_idx, batch in enumerate(dataloader):
            global_step += 1

            image = batch["image"].to(args.device)
            mask = batch["mask"].to(args.device)

            logits = model(image)

            loss = criterion(
                pred_mask=logits,
                target_mask=mask,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metrics = compute_mask_metrics(
                logits=logits,
                target=mask,
                threshold=0.5,
            )

            loss_dict = {
                "loss": float(loss.detach().cpu().item()),
                "dice": metrics["dice"],
                "iou": metrics["iou"],
            }

            last_loss_dict = loss_dict

            if global_step % args.print_freq == 0:
                print(
                    "Epoch [{}/{}] Batch [{}/{}] Step [{}] LR [{:.8f}] {}".format(
                        epoch,
                        args.epochs,
                        batch_idx + 1,
                        len(dataloader),
                        global_step,
                        args.lr,
                        format_loss_dict(loss_dict),
                    )
                )

            if global_step % args.log_freq == 0:
                append_current_loss(
                    loss_log_path=loss_log_path,
                    loss_fieldnames=loss_fieldnames,
                    epoch=epoch,
                    step=global_step,
                    lr=args.lr,
                    loss_dict=loss_dict,
                )

            if global_step == 1 or global_step % args.save_image_freq == 0:
                save_training_visuals(
                    model=model,
                    batch=batch,
                    save_dir=visual_dir,
                    epoch=epoch,
                    step=global_step,
                )

                print(
                    "Saved training visual at epoch {}, step {}".format(
                        epoch,
                        global_step,
                    )
                )

        if last_loss_dict is not None:
            append_current_loss(
                loss_log_path=loss_log_path,
                loss_fieldnames=loss_fieldnames,
                epoch=epoch,
                step=global_step,
                lr=args.lr,
                loss_dict=last_loss_dict,
            )

        latest_path = save_latest_checkpoint(
            model=model,
            optimizer=optimizer,
            checkpoint_dir=checkpoint_dir,
            args=args,
            epoch=epoch,
            step=global_step,
        )

        print("Saved latest checkpoint:", latest_path)

        if epoch % args.save_epoch_freq == 0 or epoch == args.epochs:
            epoch_path = save_epoch_checkpoint(
                model=model,
                optimizer=optimizer,
                checkpoint_dir=checkpoint_dir,
                args=args,
                epoch=epoch,
                step=global_step,
            )

            print("Saved epoch checkpoint:", epoch_path)

        print(
            "Finished epoch {}. Latest loss: {}".format(
                epoch,
                format_loss_dict(last_loss_dict)
                if last_loss_dict is not None
                else "None",
            )
        )

    print("")
    print("========== Training finished ==========")
    print("Run name:", args.run_name)
    print("Checkpoints saved to:", checkpoint_dir)
    print("Visual results saved to:", visual_dir)
    print("Loss log saved to:", loss_log_path)


if __name__ == "__main__":
    main()