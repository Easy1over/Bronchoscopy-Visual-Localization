# src/train/train_bisst.py

import argparse
import sys
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.datasets.unpaired_dataset import UnpairedBronchoscopyDataset
from src.models.bisst import BiSSTModel
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
        description="Train paper-aligned single-direction BiSST model."
    )

    # Experiment
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Experiment run name. If not set, a timestamp-based name will be created.",
    )

    # Data
    parser.add_argument(
        "--virtual_dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "processed" / "virtual"),
        help="Directory of processed virtual bronchoscopy images.",
    )
    parser.add_argument(
        "--real_dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "processed" / "real"),
        help="Directory of processed real bronchoscopy images.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Training image size.",
    )
    parser.add_argument(
        "--direction",
        type=str,
        default="virtual2real",
        choices=["virtual2real", "real2virtual"],
        help="Translation direction. virtual2real keeps the old behavior. real2virtual trains real -> virtual.",
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
        default=1,
        help="Batch size.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of dataloader workers. Use 0 on Windows.",
    )

    # Backward-compatible learning rate
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0002,
        help="Fallback initial learning rate. Used by lr_G/lr_D when they are not set.",
    )

    # TTUR / separate learning rates
    parser.add_argument(
        "--lr_G",
        type=float,
        default=None,
        help="Initial learning rate for generator-side optimizers. If None, use --lr.",
    )
    parser.add_argument(
        "--lr_D",
        type=float,
        default=None,
        help="Initial learning rate for discriminator optimizer. If None, use --lr.",
    )

    # Update frequency control
    parser.add_argument(
        "--g_update_freq",
        type=int,
        default=1,
        help="Update generator every N steps. Default 1 keeps old behavior.",
    )
    parser.add_argument(
        "--d_update_freq",
        type=int,
        default=1,
        help="Update discriminator every N steps. Use 2 or 3 to weaken D. Default 1 keeps old behavior.",
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

    # Learning rate policy
    parser.add_argument(
        "--lr_policy",
        type=str,
        default="linear",
        choices=["none", "linear", "step"],
        help="Learning rate policy.",
    )
    parser.add_argument(
        "--linear_decay_start_epoch",
        type=int,
        default=10,
        help="For linear LR decay, start decaying after this epoch.",
    )
    parser.add_argument(
        "--step_decay_epoch",
        type=int,
        default=10,
        help="For step LR decay, decay every N epochs.",
    )
    parser.add_argument(
        "--step_decay_gamma",
        type=float,
        default=0.5,
        help="For step LR decay, multiply LR by this gamma.",
    )

    # Model
    parser.add_argument(
        "--ngf",
        type=int,
        default=64,
        help="Number of generator filters.",
    )
    parser.add_argument(
        "--ndf",
        type=int,
        default=64,
        help="Number of discriminator filters.",
    )
    parser.add_argument(
        "--n_blocks",
        type=int,
        default=9,
        help="Number of ResNet blocks in generator.",
    )

    # CUT base loss weights
    parser.add_argument(
        "--lambda_gan",
        type=float,
        default=1.0,
        help="Weight for GAN loss.",
    )
    parser.add_argument(
        "--lambda_nce",
        type=float,
        default=1.0,
        help="Weight for PatchNCE loss.",
    )
    parser.add_argument(
        "--lambda_idt",
        type=float,
        default=0.5,
        help="Weight for identity PatchNCE loss.",
    )
    parser.add_argument(
        "--nce_temperature",
        type=float,
        default=0.07,
        help="Temperature for PatchNCE.",
    )
    parser.add_argument(
        "--num_patches",
        type=int,
        default=256,
        help="Number of sampled patches for PatchNCE.",
    )

    # Mask predictor Fmask
    mask_predictor_group = parser.add_mutually_exclusive_group()
    mask_predictor_group.add_argument(
        "--use_mask_predictor",
        dest="use_mask_predictor",
        action="store_true",
        help="Use pretrained / frozen mask predictor Fmask.",
    )
    mask_predictor_group.add_argument(
        "--no_mask_predictor",
        dest="use_mask_predictor",
        action="store_false",
        help="Disable mask predictor Fmask.",
    )
    parser.set_defaults(use_mask_predictor=False)

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
    parser.add_argument(
        "--mask_checkpoint",
        type=str,
        default=None,
        help="Path to pretrained mask predictor checkpoint.",
    )

    freeze_mask_group = parser.add_mutually_exclusive_group()
    freeze_mask_group.add_argument(
        "--freeze_mask_predictor",
        dest="freeze_mask_predictor",
        action="store_true",
        help="Freeze Fmask during BiSST training. This is the paper-aligned default.",
    )
    freeze_mask_group.add_argument(
        "--train_mask_predictor",
        dest="freeze_mask_predictor",
        action="store_false",
        help="Train Fmask together with G. Not paper default; only for debugging.",
    )
    parser.set_defaults(freeze_mask_predictor=True)

    # Paper-style mask loss
    parser.add_argument(
        "--lambda_mask",
        type=float,
        default=0.0,
        help="Weight for paper-style Dice mask consistency loss.",
    )
    parser.add_argument(
        "--mask_loss_type",
        type=str,
        default="dice_consistency",
        help="Mask loss type. Paper-aligned default: dice_consistency.",
    )

    # Embedding extractor Eh
    embedding_group = parser.add_mutually_exclusive_group()
    embedding_group.add_argument(
        "--use_embedding_extractor",
        dest="use_embedding_extractor",
        action="store_true",
        help="Use structure embedding extractor Eh.",
    )
    embedding_group.add_argument(
        "--no_embedding_extractor",
        dest="use_embedding_extractor",
        action="store_false",
        help="Disable structure embedding extractor Eh.",
    )
    parser.set_defaults(use_embedding_extractor=True)

    parser.add_argument(
        "--embedding_extractor_type",
        type=str,
        default="conv",
        choices=["conv", "pool"],
        help="Embedding extractor type.",
    )
    parser.add_argument(
        "--embedding_input_channels",
        type=int,
        default=None,
        help="Input channels of selected structural feature. If None, use ngf * 4.",
    )
    parser.add_argument(
        "--embedding_hidden_channels",
        type=int,
        default=128,
        help="Hidden channels of embedding extractor.",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=128,
        help="Output embedding dimension. Paper uses 128.",
    )

    # Paper-style structural semantic loss
    parser.add_argument(
        "--lambda_semantic",
        type=float,
        default=0.0,
        help="Weight for structural semantic consistency loss.",
    )
    parser.add_argument(
        "--lambda_semantic_neg",
        type=float,
        default=0.1,
        help="Weight for negative separation term inside structural semantic loss.",
    )
    parser.add_argument(
        "--semantic_loss_type",
        type=str,
        default="structural",
        help="Semantic loss type. Paper-aligned default: structural.",
    )

    detach_semantic_weight_group = parser.add_mutually_exclusive_group()
    detach_semantic_weight_group.add_argument(
        "--detach_semantic_negative_weight",
        dest="detach_semantic_negative_weight",
        action="store_true",
        help="Detach mask-aware negative weight in semantic loss.",
    )
    detach_semantic_weight_group.add_argument(
        "--no_detach_semantic_negative_weight",
        dest="detach_semantic_negative_weight",
        action="store_false",
        help="Do not detach mask-aware negative weight.",
    )
    parser.set_defaults(detach_semantic_negative_weight=True)

    # Logging and saving
    parser.add_argument(
        "--print_freq",
        type=int,
        default=50,
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
        default=200,
        help="Save visual results every N iterations.",
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
        default=str(PROJECT_ROOT / "checkpoints" / "bisst"),
        help="Root directory for BiSST checkpoints.",
    )
    parser.add_argument(
        "--result_root",
        type=str,
        default=str(PROJECT_ROOT / "results" / "bisst"),
        help="Root directory for BiSST results.",
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


def finalize_args(args):
    """
    Fill backward-compatible defaults and sanitize update frequencies.
    """
    if args.lr_G is None:
        args.lr_G = args.lr

    if args.lr_D is None:
        args.lr_D = args.lr

    args.g_update_freq = max(1, int(args.g_update_freq))
    args.d_update_freq = max(1, int(args.d_update_freq))

    return args


def format_loss_dict(loss_dict: Dict[str, float]) -> str:
    parts = []

    for key, value in loss_dict.items():
        parts.append("{}: {:.4f}".format(key, value))

    return " | ".join(parts)


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
        f.write("virtual_dir: {}\n".format(args.virtual_dir))
        f.write("real_dir: {}\n".format(args.real_dir))
        f.write("image_size: {}\n".format(args.image_size))
        f.write("direction: {}\n".format(args.direction))

        f.write("\n[Training]\n")
        f.write("epochs: {}\n".format(args.epochs))
        f.write("batch_size: {}\n".format(args.batch_size))
        f.write("num_workers: {}\n".format(args.num_workers))
        f.write("lr: {}\n".format(args.lr))
        f.write("lr_G: {}\n".format(args.lr_G))
        f.write("lr_D: {}\n".format(args.lr_D))
        f.write("g_update_freq: {}\n".format(args.g_update_freq))
        f.write("d_update_freq: {}\n".format(args.d_update_freq))
        f.write("beta1: {}\n".format(args.beta1))
        f.write("beta2: {}\n".format(args.beta2))
        f.write("lr_policy: {}\n".format(args.lr_policy))
        f.write("linear_decay_start_epoch: {}\n".format(args.linear_decay_start_epoch))
        f.write("step_decay_epoch: {}\n".format(args.step_decay_epoch))
        f.write("step_decay_gamma: {}\n".format(args.step_decay_gamma))

        f.write("\n[Model]\n")
        f.write("ngf: {}\n".format(args.ngf))
        f.write("ndf: {}\n".format(args.ndf))
        f.write("n_blocks: {}\n".format(args.n_blocks))

        f.write("\n[CUT losses]\n")
        f.write("lambda_gan: {}\n".format(args.lambda_gan))
        f.write("lambda_nce: {}\n".format(args.lambda_nce))
        f.write("lambda_idt: {}\n".format(args.lambda_idt))
        f.write("nce_temperature: {}\n".format(args.nce_temperature))
        f.write("num_patches: {}\n".format(args.num_patches))

        f.write("\n[Mask predictor Fmask]\n")
        f.write("use_mask_predictor: {}\n".format(args.use_mask_predictor))
        f.write("mask_predictor_type: {}\n".format(args.mask_predictor_type))
        f.write("mask_base_channels: {}\n".format(args.mask_base_channels))
        f.write("mask_checkpoint: {}\n".format(args.mask_checkpoint))
        f.write("freeze_mask_predictor: {}\n".format(args.freeze_mask_predictor))
        f.write("lambda_mask: {}\n".format(args.lambda_mask))
        f.write("mask_loss_type: {}\n".format(args.mask_loss_type))

        f.write("\n[Embedding extractor Eh]\n")
        f.write("use_embedding_extractor: {}\n".format(args.use_embedding_extractor))
        f.write("embedding_extractor_type: {}\n".format(args.embedding_extractor_type))
        f.write("embedding_input_channels: {}\n".format(args.embedding_input_channels))
        f.write("embedding_hidden_channels: {}\n".format(args.embedding_hidden_channels))
        f.write("embedding_dim: {}\n".format(args.embedding_dim))

        f.write("\n[Semantic loss]\n")
        f.write("lambda_semantic: {}\n".format(args.lambda_semantic))
        f.write("lambda_semantic_neg: {}\n".format(args.lambda_semantic_neg))
        f.write("semantic_loss_type: {}\n".format(args.semantic_loss_type))
        f.write("detach_semantic_negative_weight: {}\n".format(args.detach_semantic_negative_weight))

        f.write("\n[Paths]\n")
        f.write("device: {}\n".format(args.device))
        f.write("checkpoint_dir: {}\n".format(checkpoint_dir))
        f.write("result_dir: {}\n".format(result_dir))
        f.write("visual_dir: {}\n".format(visual_dir))
        f.write("resume: {}\n".format(args.resume))


def save_training_visuals(
    model: BiSSTModel,
    save_dir: Path,
    epoch: int,
    step: int,
) -> None:
    """
    Save visual comparison:
        real_A | fake_B | real_B | idt_B(optional)

    Direction meaning:
        virtual2real:
            virtual input | fake real | real target | idt real(optional)

        real2virtual:
            real input | fake virtual | virtual target | idt virtual(optional)
    """
    visuals = model.get_current_visuals()

    real_A = visuals["real_A"][0]
    fake_B = visuals["fake_B"][0]
    real_B = visuals["real_B"][0]

    image_list = [real_A, fake_B, real_B]

    if "idt_B" in visuals:
        idt_B = visuals["idt_B"][0]
        image_list.append(idt_B)

    save_path = save_dir / "epoch_{:03d}_step_{:07d}.png".format(
        epoch,
        step,
    )

    save_image_grid(
        image_tensors=image_list,
        save_path=save_path,
        padding=5,
    )


def append_current_loss(
    loss_log_path: Path,
    loss_fieldnames,
    epoch: int,
    step: int,
    lr_G: float,
    lr_D: float,
    update_G: bool,
    update_D: bool,
    loss_dict: Dict[str, float],
) -> None:
    """
    Append one row to loss CSV.

    Missing fields will be filled by 0.0 to keep CSV stable.
    """
    log_row = {}

    for field in loss_fieldnames:
        log_row[field] = 0.0

    log_row["epoch"] = epoch
    log_row["step"] = step

    # Keep old field name "lr" for backward compatibility.
    # It now records generator-side learning rate.
    log_row["lr"] = lr_G
    log_row["lr_G"] = lr_G
    log_row["lr_D"] = lr_D
    log_row["update_G"] = int(update_G)
    log_row["update_D"] = int(update_D)

    for key, value in loss_dict.items():
        if key in log_row:
            log_row[key] = value

    append_loss_log(
        save_path=loss_log_path,
        row=log_row,
        fieldnames=loss_fieldnames,
    )


def build_checkpoint_extra(args):
    return {
        "run_name": args.run_name,
        "direction": args.direction,
        "args": vars(args),
    }


def save_latest_checkpoint(
    model: BiSSTModel,
    checkpoint_dir: Path,
    args,
    epoch: int,
    step: int,
) -> Path:
    latest_path = checkpoint_dir / "latest.pt"

    save_checkpoint(
        save_path=latest_path,
        models=model.get_model_dict(),
        optimizers=model.get_optimizer_dict(),
        epoch=epoch,
        step=step,
        extra=build_checkpoint_extra(args),
    )

    return latest_path


def save_epoch_checkpoint(
    model: BiSSTModel,
    checkpoint_dir: Path,
    args,
    epoch: int,
    step: int,
) -> Path:
    epoch_path = checkpoint_dir / "epoch_{:03d}.pt".format(epoch)

    save_checkpoint(
        save_path=epoch_path,
        models=model.get_model_dict(),
        optimizers=model.get_optimizer_dict(),
        epoch=epoch,
        step=step,
        extra=build_checkpoint_extra(args),
    )

    return epoch_path


def load_training_state(
    model: BiSSTModel,
    resume_path: str,
    device: str,
):
    """
    Load model and optimizer states.
    """
    checkpoint_path = Path(resume_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Resume checkpoint does not exist: {}".format(checkpoint_path)
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if "models" not in checkpoint:
        raise KeyError("Checkpoint does not contain key 'models'.")

    saved_models = checkpoint["models"]
    current_models = model.get_model_dict()

    for name, net in current_models.items():
        if name in saved_models:
            net.load_state_dict(
                saved_models[name],
                strict=True,
            )
        else:
            print("Warning: model key '{}' not found in checkpoint.".format(name))

    if "optimizers" in checkpoint:
        saved_optimizers = checkpoint["optimizers"]
        current_optimizers = model.get_optimizer_dict()

        for name, optimizer in current_optimizers.items():
            if name in saved_optimizers:
                optimizer.load_state_dict(saved_optimizers[name])
            else:
                print("Warning: optimizer key '{}' not found in checkpoint.".format(name))
    else:
        print("Warning: checkpoint does not contain optimizers. Optimizers are not resumed.")

    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    global_step = int(checkpoint.get("step", 0))

    return start_epoch, global_step


def get_lr_scale(
    epoch: int,
    args,
) -> float:
    """
    Compute learning rate scale by epoch.

    epoch is 1-based.
    """
    if args.lr_policy == "none":
        return 1.0

    if args.lr_policy == "linear":
        if epoch < args.linear_decay_start_epoch:
            return 1.0

        decay_total = max(1, args.epochs - args.linear_decay_start_epoch + 1)
        decay_progress = epoch - args.linear_decay_start_epoch + 1

        scale = 1.0 - decay_progress / decay_total
        scale = max(0.0, scale)

        return scale

    if args.lr_policy == "step":
        num_decays = (epoch - 1) // max(1, args.step_decay_epoch)
        scale = args.step_decay_gamma ** num_decays

        return scale

    raise ValueError("Unsupported lr_policy: {}".format(args.lr_policy))


def set_optimizer_lrs(
    model: BiSSTModel,
    lr_G: float,
    lr_D: float,
) -> None:
    """
    Set separate learning rates for generator-side and discriminator optimizers.

    Expected optimizer names from model.get_optimizer_dict():
        "G" for generator-side optimizer
        "D" for discriminator optimizer

    For safety, every optimizer whose name is not exactly "D" uses lr_G.
    This keeps compatibility with possible optimizers such as F, Eh, Fmask, etc.
    """
    optimizers = model.get_optimizer_dict()

    for name, optimizer in optimizers.items():
        if name == "D":
            target_lr = lr_D
        else:
            target_lr = lr_G

        for param_group in optimizer.param_groups:
            param_group["lr"] = target_lr


def get_current_lrs(
    model: BiSSTModel,
):
    """
    Return current generator-side and discriminator learning rates.
    """
    optimizers = model.get_optimizer_dict()

    lr_G = None
    lr_D = None

    if "G" in optimizers:
        lr_G = optimizers["G"].param_groups[0]["lr"]

    if "D" in optimizers:
        lr_D = optimizers["D"].param_groups[0]["lr"]

    return lr_G, lr_D


def get_current_lr(
    model: BiSSTModel,
) -> float:
    """
    Backward-compatible helper.

    Return generator-side LR if available.
    """
    lr_G, _ = get_current_lrs(model)

    if lr_G is None:
        return 0.0

    return lr_G


def compute_update_flags(
    global_step: int,
    args,
):
    """
    Decide whether to update G and D at this step.

    Default:
        g_update_freq = 1
        d_update_freq = 1

    Then both G and D are updated every step, which matches old behavior.
    """
    update_G = (global_step % args.g_update_freq == 0)
    update_D = (global_step % args.d_update_freq == 0)

    return update_G, update_D


def optimize_model_parameters(
    model: BiSSTModel,
    update_G: bool,
    update_D: bool,
):
    """
    Compatibility wrapper.

    After src/models/bisst.py is updated, BiSSTModel.optimize_parameters()
    should support:
        optimize_parameters(update_G=True, update_D=True)

    Before that update, this wrapper still allows old behavior only when
    both update_G and update_D are True.
    """
    try:
        return model.optimize_parameters(
            update_G=update_G,
            update_D=update_D,
        )
    except TypeError as exc:
        if update_G and update_D:
            return model.optimize_parameters()

        raise TypeError(
            "BiSSTModel.optimize_parameters() does not yet support "
            "update_G/update_D. Please update src/models/bisst.py before using "
            "g_update_freq or d_update_freq values other than 1."
        ) from exc


def main():
    args = parse_args()
    args = finalize_args(args)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Fall back to CPU.")
        args.device = "cpu"

    if args.lambda_mask > 0 and not args.use_mask_predictor:
        print(
            "Warning: lambda_mask > 0 but use_mask_predictor=False. "
            "Mask loss will be zero."
        )

    if args.lambda_semantic > 0 and not args.use_mask_predictor:
        print(
            "Warning: lambda_semantic > 0 but use_mask_predictor=False. "
            "Current paper-style structural semantic loss requires Fmask "
            "for mask-aware negative weighting, so semantic loss will be zero."
        )

    if args.use_mask_predictor and args.mask_checkpoint is None:
        print(
            "Warning: use_mask_predictor=True but mask_checkpoint=None. "
            "Fmask will be randomly initialized. This is only useful for "
            "debugging the training pipeline, not for paper-aligned training."
        )

    args.run_name = create_run_name(
        prefix="bisst_{}".format(args.direction),
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
        "lr_G",
        "lr_D",
        "update_G",
        "update_D",
        "loss_D",
        "loss_G_total",
        "loss_G_GAN",
        "loss_NCE",
        "loss_IDT",
        "loss_mask",
        "loss_semantic",
        "loss_sem_pos",
        "loss_sem_neg",
    ]

    init_loss_log(
        save_path=loss_log_path,
        fieldnames=loss_fieldnames,
    )

    print("========== Experiment ==========")
    print("Run name:", args.run_name)
    print("Project root:", PROJECT_ROOT)
    print("Virtual dir:", args.virtual_dir)
    print("Real dir:", args.real_dir)
    print("Direction:", args.direction)
    print("Checkpoint dir:", checkpoint_dir)
    print("Result dir:", result_dir)
    print("Visual dir:", visual_dir)
    print("Config saved to:", config_path)
    print("Run info saved to:", run_info_path)
    print("Loss log saved to:", loss_log_path)
    print("Device:", args.device)
    print("Base lr:", args.lr)
    print("Initial lr_G:", args.lr_G)
    print("Initial lr_D:", args.lr_D)
    print("G update freq:", args.g_update_freq)
    print("D update freq:", args.d_update_freq)
    print("LR policy:", args.lr_policy)

    dataset = UnpairedBronchoscopyDataset(
        virtual_dir=args.virtual_dir,
        real_dir=args.real_dir,
        image_size=args.image_size,
        train=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    print("")
    print("========== Data ==========")
    print("Dataset size:", len(dataset))
    print("Number of batches per epoch:", len(dataloader))

    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty. Please check virtual_dir and real_dir.")

    if len(dataloader) == 0:
        raise RuntimeError(
            "Dataloader has 0 batches. "
            "This may happen when dataset size is smaller than batch_size with drop_last=True."
        )

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

    start_epoch = 1
    global_step = 0

    if args.resume is not None:
        print("")
        print("========== Resume training ==========")
        print("Resume checkpoint:", args.resume)

        start_epoch, global_step = load_training_state(
            model=model,
            resume_path=args.resume,
            device=args.device,
        )

        print("Resume from epoch:", start_epoch)
        print("Resume global step:", global_step)

    for epoch in range(start_epoch, args.epochs + 1):
        lr_scale = get_lr_scale(
            epoch=epoch,
            args=args,
        )

        current_lr_G = args.lr_G * lr_scale
        current_lr_D = args.lr_D * lr_scale

        set_optimizer_lrs(
            model=model,
            lr_G=current_lr_G,
            lr_D=current_lr_D,
        )

        print("")
        print("========== Epoch {}/{} ==========".format(epoch, args.epochs))
        print("Current LR G:", current_lr_G)
        print("Current LR D:", current_lr_D)

        last_loss_dict = None
        last_update_G = True
        last_update_D = True

        for batch_idx, batch in enumerate(dataloader):
            global_step += 1

            update_G, update_D = compute_update_flags(
                global_step=global_step,
                args=args,
            )

            last_update_G = update_G
            last_update_D = update_D

            model.set_input(batch)

            loss_dict = optimize_model_parameters(
                model=model,
                update_G=update_G,
                update_D=update_D,
            )

            last_loss_dict = loss_dict

            if global_step % args.print_freq == 0:
                print(
                    "Epoch [{}/{}] Batch [{}/{}] Step [{}] "
                    "LR_G [{:.8f}] LR_D [{:.8f}] "
                    "update_G [{}] update_D [{}] {}".format(
                        epoch,
                        args.epochs,
                        batch_idx + 1,
                        len(dataloader),
                        global_step,
                        current_lr_G,
                        current_lr_D,
                        int(update_G),
                        int(update_D),
                        format_loss_dict(loss_dict),
                    )
                )

            if global_step % args.log_freq == 0:
                append_current_loss(
                    loss_log_path=loss_log_path,
                    loss_fieldnames=loss_fieldnames,
                    epoch=epoch,
                    step=global_step,
                    lr_G=current_lr_G,
                    lr_D=current_lr_D,
                    update_G=update_G,
                    update_D=update_D,
                    loss_dict=loss_dict,
                )

            if global_step == 1 or global_step % args.save_image_freq == 0:
                save_training_visuals(
                    model=model,
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
                lr_G=current_lr_G,
                lr_D=current_lr_D,
                update_G=last_update_G,
                update_D=last_update_D,
                loss_dict=last_loss_dict,
            )

        latest_path = save_latest_checkpoint(
            model=model,
            checkpoint_dir=checkpoint_dir,
            args=args,
            epoch=epoch,
            step=global_step,
        )

        print("Saved latest checkpoint:", latest_path)

        if epoch % args.save_epoch_freq == 0 or epoch == args.epochs:
            epoch_path = save_epoch_checkpoint(
                model=model,
                checkpoint_dir=checkpoint_dir,
                args=args,
                epoch=epoch,
                step=global_step,
            )

            print("Saved epoch checkpoint:", epoch_path)

        save_training_visuals(
            model=model,
            save_dir=visual_dir,
            epoch=epoch,
            step=global_step,
        )

        print(
            "Finished epoch {}. Latest loss: {}".format(
                epoch,
                format_loss_dict(last_loss_dict)
                if last_loss_dict is not None
                else "None",
            )
        )

    final_lr_G, final_lr_D = get_current_lrs(model)

    print("")
    print("========== Training finished ==========")
    print("Run name:", args.run_name)
    print("Direction:", args.direction)
    print("Checkpoints saved to:", checkpoint_dir)
    print("Visual results saved to:", visual_dir)
    print("Loss log saved to:", loss_log_path)
    print("Final LR G:", final_lr_G)
    print("Final LR D:", final_lr_D)


if __name__ == "__main__":
    main()


# virtual -> real, no mask / semantic
# python src/train/train_bisst.py --direction virtual2real --epochs 1 --batch_size 1 --print_freq 1 --log_freq 1 --save_image_freq 20

# real -> virtual, no mask / semantic
# python src/train/train_bisst.py --direction real2virtual --epochs 1 --batch_size 1 --print_freq 1 --log_freq 1 --save_image_freq 20

# mask predictor training
# python src/train/train_mask_predictor.py --epochs 1 --batch_size 2 --print_freq 1 --log_freq 1 --save_image_freq 5 --image_dir data/masks/images --mask_dir data/masks/masks

# virtual -> real, with mask / semantic
# python src/train/train_bisst.py --direction virtual2real --epochs 1 --batch_size 2 --print_freq 1 --log_freq 1 --save_image_freq 20 --use_mask_predictor --freeze_mask_predictor --mask_checkpoint checkpoints/mask_predictor/20260617_121050_mask_predictor/latest.pt --lambda_mask 0.1 --lambda_semantic 0.1 --lambda_semantic_neg 0.1

# real -> virtual, with mask / semantic
# python src/train/train_bisst.py --direction real2virtual --epochs 1 --batch_size 2 --print_freq 1 --log_freq 1 --save_image_freq 20 --use_mask_predictor --freeze_mask_predictor --mask_checkpoint checkpoints/mask_predictor/20260617_121050_mask_predictor/latest.pt --lambda_mask 0.1 --lambda_semantic 0.1 --lambda_semantic_neg 0.1
