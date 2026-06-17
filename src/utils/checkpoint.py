from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.optim as optim


def save_checkpoint(
    save_path: Union[str, Path],
    models: Dict[str, nn.Module],
    optimizers: Optional[Dict[str, optim.Optimizer]] = None,
    epoch: int = 0,
    step: int = 0,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save model and optimizer states.

    Args:
        save_path: checkpoint path
        models: dict of model_name -> model
        optimizers: dict of optimizer_name -> optimizer
        epoch: current epoch
        step: current global step
        extra: extra information to save
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "step": step,
        "models": {},
        "optimizers": {},
        "extra": extra if extra is not None else {},
    }

    for name, model in models.items():
        checkpoint["models"][name] = model.state_dict()

    if optimizers is not None:
        for name, optimizer in optimizers.items():
            checkpoint["optimizers"][name] = optimizer.state_dict()

    torch.save(checkpoint, save_path)


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    models: Optional[Dict[str, nn.Module]] = None,
    optimizers: Optional[Dict[str, optim.Optimizer]] = None,
    map_location: Union[str, torch.device] = "cpu",
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Load model and optimizer states.

    Args:
        checkpoint_path: checkpoint path
        models: dict of model_name -> model
        optimizers: dict of optimizer_name -> optimizer
        map_location: torch load map location
        strict: whether to strictly load model state dict

    Returns:
        checkpoint dict
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError("Checkpoint not found: {}".format(checkpoint_path))

    checkpoint = torch.load(
        checkpoint_path,
        map_location=map_location,
    )

    if models is not None:
        saved_models = checkpoint.get("models", {})

        for name, model in models.items():
            if name not in saved_models:
                raise KeyError(
                    "Model '{}' not found in checkpoint. Available models: {}".format(
                        name,
                        list(saved_models.keys()),
                    )
                )
            model.load_state_dict(saved_models[name], strict=strict)

    if optimizers is not None:
        saved_optimizers = checkpoint.get("optimizers", {})

        for name, optimizer in optimizers.items():
            if name not in saved_optimizers:
                raise KeyError(
                    "Optimizer '{}' not found in checkpoint. Available optimizers: {}".format(
                        name,
                        list(saved_optimizers.keys()),
                    )
                )
            optimizer.load_state_dict(saved_optimizers[name])

    return checkpoint


def get_checkpoint_info(
    checkpoint_path: Union[str, Path],
) -> Dict[str, Any]:
    """
    Load only checkpoint metadata.
    """
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    info = {
        "epoch": checkpoint.get("epoch", None),
        "step": checkpoint.get("step", None),
        "model_names": list(checkpoint.get("models", {}).keys()),
        "optimizer_names": list(checkpoint.get("optimizers", {}).keys()),
        "extra": checkpoint.get("extra", {}),
    }

    return info


if __name__ == "__main__":
    from src.models.generator import ResnetGenerator

    model = ResnetGenerator(
        input_nc=3,
        output_nc=3,
        ngf=32,
        n_blocks=3,
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.0002,
        betas=(0.5, 0.999),
    )

    save_path = Path("checkpoints") / "debug" / "checkpoint_test.pt"

    save_checkpoint(
        save_path=save_path,
        models={"G": model},
        optimizers={"G": optimizer},
        epoch=1,
        step=100,
        extra={"note": "checkpoint test"},
    )

    info = get_checkpoint_info(save_path)
    print("Checkpoint test")
    print(info)

    load_checkpoint(
        checkpoint_path=save_path,
        models={"G": model},
        optimizers={"G": optimizer},
        map_location="cpu",
    )

    print("Loaded checkpoint from:", save_path)