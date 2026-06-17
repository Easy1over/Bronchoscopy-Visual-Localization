import argparse
import sys
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.datasets.unpaired_dataset import UnpairedBronchoscopyDataset
from src.models.cut import CUTModel
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
        description="Train minimal CUT baseline for bronchoscopy style translation."
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
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0002,
        help="Learning rate.",
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

    # Loss weights
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
        default=str(PROJECT_ROOT / "checkpoints" / "cut"),
        help="Root directory for CUT checkpoints.",
    )
    parser.add_argument(
        "--result_root",
        type=str,
        default=str(PROJECT_ROOT / "results" / "cut"),
        help="Root directory for CUT results.",
    )

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


def save_run_info(
    save_path: Path,
    args,
    checkpoint_dir: Path,
    result_dir: Path,
    visual_dir: Path,
) -> None:
    """
    Save a lightweight text summary for quick human inspection.
    This is separate from train_config.txt.
    """

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("run_name: {}\n".format(args.run_name))
        f.write("project_root: {}\n".format(PROJECT_ROOT))
        f.write("virtual_dir: {}\n".format(args.virtual_dir))
        f.write("real_dir: {}\n".format(args.real_dir))
        f.write("image_size: {}\n".format(args.image_size))
        f.write("epochs: {}\n".format(args.epochs))
        f.write("batch_size: {}\n".format(args.batch_size))
        f.write("lr: {}\n".format(args.lr))
        f.write("device: {}\n".format(args.device))
        f.write("checkpoint_dir: {}\n".format(checkpoint_dir))
        f.write("result_dir: {}\n".format(result_dir))
        f.write("visual_dir: {}\n".format(visual_dir))


def save_training_visuals(
    model: CUTModel,
    save_dir: Path,
    epoch: int,
    step: int,
) -> None:
    """
    Save visual comparison:
        real_A | fake_B | real_B | idt_B(optional)

    real_A: virtual input
    fake_B: generated real-style image
    real_B: real target-domain image
    idt_B: identity output for real image
    """

    visuals = model.get_current_visuals()

    real_A = visuals["real_A"][0]
    fake_B = visuals["fake_B"][0]
    real_B = visuals["real_B"][0]

    image_list = [real_A, fake_B, real_B]

    if "idt_B" in visuals:
        idt_B = visuals["idt_B"][0]
        image_list.append(idt_B)

    save_path = save_dir / "epoch_{:03d}_step_{:07d}.png".format(epoch, step)

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
    loss_dict: Dict[str, float],
) -> None:
    """
    Append one row to loss csv.
    """

    log_row = {
        "epoch": epoch,
        "step": step,
    }
    log_row.update(loss_dict)

    append_loss_log(
        save_path=loss_log_path,
        row=log_row,
        fieldnames=loss_fieldnames,
    )


def build_checkpoint_extra(args):
    """
    Save complete training arguments into checkpoint.
    This makes test/inference scripts easier to reproduce.
    """

    return {
        "run_name": args.run_name,
        "args": vars(args),
    }


def save_latest_checkpoint(
    model: CUTModel,
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
    model: CUTModel,
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


def main():
    args = parse_args()

    # Device fallback
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Fall back to CPU.")
        args.device = "cpu"

    # Create independent experiment folder
    args.run_name = create_run_name(
        prefix="cut",
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
        "loss_D",
        "loss_G_total",
        "loss_G_GAN",
        "loss_NCE",
        "loss_IDT",
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
    print("Checkpoint dir:", checkpoint_dir)
    print("Result dir:", result_dir)
    print("Visual dir:", visual_dir)
    print("Config saved to:", config_path)
    print("Run info saved to:", run_info_path)
    print("Loss log saved to:", loss_log_path)
    print("Device:", args.device)

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
        raise RuntimeError(
            "Dataset is empty. Please check virtual_dir and real_dir."
        )

    if len(dataloader) == 0:
        raise RuntimeError(
            "Dataloader has 0 batches. "
            "This may happen when dataset size is smaller than batch_size with drop_last=True."
        )

    model = CUTModel(
        input_nc=3,
        output_nc=3,
        ngf=args.ngf,
        ndf=args.ndf,
        n_blocks=args.n_blocks,
        lr=args.lr,
        lambda_gan=args.lambda_gan,
        lambda_nce=args.lambda_nce,
        lambda_idt=args.lambda_idt,
        device=args.device,
    )

    global_step = 0

    for epoch in range(1, args.epochs + 1):
        print("")
        print("========== Epoch {}/{} ==========".format(epoch, args.epochs))

        last_loss_dict = None

        for batch_idx, batch in enumerate(dataloader):
            global_step += 1

            model.set_input(batch)
            loss_dict = model.optimize_parameters()
            last_loss_dict = loss_dict

            if global_step % args.print_freq == 0:
                print(
                    "Epoch [{}/{}] Batch [{}/{}] Step [{}] {}".format(
                        epoch,
                        args.epochs,
                        batch_idx + 1,
                        len(dataloader),
                        global_step,
                        format_loss_dict(loss_dict),
                    )
                )

            if global_step % args.log_freq == 0:
                append_current_loss(
                    loss_log_path=loss_log_path,
                    loss_fieldnames=loss_fieldnames,
                    epoch=epoch,
                    step=global_step,
                    loss_dict=loss_dict,
                )

            # Save first visual immediately, then save every save_image_freq steps.
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

        # Ensure every epoch has at least one loss record.
        if last_loss_dict is not None:
            append_current_loss(
                loss_log_path=loss_log_path,
                loss_fieldnames=loss_fieldnames,
                epoch=epoch,
                step=global_step,
                loss_dict=last_loss_dict,
            )

        # Always save latest checkpoint at the end of every epoch.
        latest_path = save_latest_checkpoint(
            model=model,
            checkpoint_dir=checkpoint_dir,
            args=args,
            epoch=epoch,
            step=global_step,
        )

        print("Saved latest checkpoint:", latest_path)

        # Save epoch checkpoint at selected epochs.
        if epoch % args.save_epoch_freq == 0 or epoch == args.epochs:
            epoch_path = save_epoch_checkpoint(
                model=model,
                checkpoint_dir=checkpoint_dir,
                args=args,
                epoch=epoch,
                step=global_step,
            )

            print("Saved epoch checkpoint:", epoch_path)

        # Save one visual result at the end of every epoch.
        save_training_visuals(
            model=model,
            save_dir=visual_dir,
            epoch=epoch,
            step=global_step,
        )

        print(
            "Finished epoch {}. Latest loss: {}".format(
                epoch,
                format_loss_dict(last_loss_dict) if last_loss_dict is not None else "None",
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