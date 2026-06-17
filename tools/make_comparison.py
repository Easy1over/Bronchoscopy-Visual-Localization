import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.datasets.unpaired_dataset import UnpairedBronchoscopyDataset
from src.utils.visualize import save_tensor_image, save_image_grid


def main():
    dataset = UnpairedBronchoscopyDataset(
        virtual_dir=PROJECT_ROOT / "data" / "processed" / "virtual",
        real_dir=PROJECT_ROOT / "data" / "processed" / "real",
        image_size=256,
        train=False,
    )

    output_dir = PROJECT_ROOT / "results" / "dataset_check"
    output_dir.mkdir(parents=True, exist_ok=True)

    num_samples = min(5, len(dataset))

    for i in range(num_samples):
        sample = dataset[i]

        virtual = sample["virtual"]
        real = sample["real"]

        save_tensor_image(
            virtual,
            output_dir / "sample_{:03d}_virtual.png".format(i)
        )

        save_tensor_image(
            real,
            output_dir / "sample_{:03d}_real.png".format(i)
        )

        save_image_grid(
            [virtual, real],
            output_dir / "sample_{:03d}_comparison.png".format(i)
        )

        print("Saved sample {}".format(i))
        print("  virtual:", sample["virtual_path"])
        print("  real:   ", sample["real_path"])

    print("Done. Results saved to:", output_dir)


if __name__ == "__main__":
    main()