import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union


def create_run_name(
    prefix: str = "run",
    run_name: Optional[str] = None,
) -> str:
    """
    Create a run name.

    If run_name is given, use it directly.
    Otherwise, create a timestamp-based name.

    Example:
        20260616_193012_cut
    """
    if run_name is not None and run_name.strip() != "":
        return run_name.strip()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return "{}_{}".format(timestamp, prefix)


def setup_experiment_dirs(
    checkpoint_root: Union[str, Path],
    result_root: Union[str, Path],
    run_name: str,
) -> Dict[str, Path]:
    """
    Create experiment directories.

    Directory structure:
        checkpoints/cut/<run_name>/
        results/cut/<run_name>/
            comparison/
            translated/
            metrics/
    """
    checkpoint_root = Path(checkpoint_root)
    result_root = Path(result_root)

    checkpoint_dir = checkpoint_root / run_name
    result_dir = result_root / run_name

    visual_dir = result_dir / "comparison"
    translated_dir = result_dir / "translated"
    metrics_dir = result_dir / "metrics"

    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    visual_dir.mkdir(parents=True, exist_ok=True)
    translated_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    return {
        "checkpoint_dir": checkpoint_dir,
        "result_dir": result_dir,
        "visual_dir": visual_dir,
        "translated_dir": translated_dir,
        "metrics_dir": metrics_dir,
    }


def save_config(
    args: Any,
    save_path: Union[str, Path],
) -> None:
    """
    Save training arguments to a text file.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    args_dict = vars(args)

    with open(save_path, "w", encoding="utf-8") as f:
        for key in sorted(args_dict.keys()):
            f.write("{}: {}\n".format(key, args_dict[key]))


def init_loss_log(
    save_path: Union[str, Path],
    fieldnames,
) -> None:
    """
    Initialize a CSV loss log file.

    Args:
        save_path: CSV file path
        fieldnames: list of column names
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def append_loss_log(
    save_path: Union[str, Path],
    row: Dict[str, Any],
    fieldnames,
) -> None:
    """
    Append one row to CSV loss log.
    """
    save_path = Path(save_path)

    with open(save_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        clean_row = {}
        for key in fieldnames:
            clean_row[key] = row.get(key, "")

        writer.writerow(clean_row)


def save_text(
    text: str,
    save_path: Union[str, Path],
) -> None:
    """
    Save plain text to disk.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(text)