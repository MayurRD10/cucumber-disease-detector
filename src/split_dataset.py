"""
Splits a flat folder-of-classes download (e.g. the Mendeley Cucumber
Disease Recognition dataset, which ships as one folder per class) into the
train/val/test layout that dataset.py expects.

Usage:
    python src/split_dataset.py --source raw_download/ --dest data/ \
        --train 0.7 --val 0.15 --test 0.15

Source layout expected:
    raw_download/Anthracnose/*.jpg
    raw_download/Bacterial Wilt/*.jpg
    ...

Produces:
    data/train/Anthracnose/*.jpg
    data/val/Anthracnose/*.jpg
    data/test/Anthracnose/*.jpg
    ...
(class folder names are slugified: spaces -> underscores)
"""

import argparse
import platform
import random
import shutil
from pathlib import Path


def slugify(name: str) -> str:
    return name.strip().replace(" ", "_").replace("-", "_")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="Folder containing one subfolder per class")
    p.add_argument("--dest", default="data", help="Output folder (will contain train/val/test)")
    p.add_argument("--train", type=float, default=0.7)
    p.add_argument("--val", type=float, default=0.15)
    p.add_argument("--test", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    # Default to copying on Windows where symlinks need admin / Developer Mode.
    is_windows = platform.system() == "Windows"
    p.add_argument(
        "--copy", action="store_true", default=is_windows,
        help="Copy files instead of symlinking "
             "(default: True on Windows, False elsewhere).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    assert abs(args.train + args.val + args.test - 1.0) < 1e-6, "splits must sum to 1.0"

    random.seed(args.seed)
    source = Path(args.source)
    dest = Path(args.dest)

    class_dirs = sorted([d for d in source.iterdir() if d.is_dir()])
    if not class_dirs:
        raise SystemExit(f"No class subfolders found in {source}")

    summary = {}
    for class_dir in class_dirs:
        class_name = slugify(class_dir.name)
        images = sorted(
            [f for f in class_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
        )
        random.shuffle(images)

        n = len(images)
        n_train = int(n * args.train)
        n_val = int(n * args.val)
        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }

        for split_name, files in splits.items():
            split_class_dir = dest / split_name / class_name
            split_class_dir.mkdir(parents=True, exist_ok=True)
            for f in files:
                target = split_class_dir / f.name
                if target.exists():
                    continue
                if args.copy:
                    shutil.copy2(f, target)
                else:
                    try:
                        target.symlink_to(f.resolve())
                    except OSError as exc:
                        raise SystemExit(
                            f"Symlink creation failed: {exc}\n"
                            f"On Windows, run with --copy, or enable "
                            f"Developer Mode in Settings > For Developers."
                        ) from exc

        summary[class_name] = {k: len(v) for k, v in splits.items()}

    print("Split complete:")
    print(f"{'class':30}{'train':>8}{'val':>8}{'test':>8}")
    for cname, counts in summary.items():
        print(f"{cname:30}{counts['train']:>8}{counts['val']:>8}{counts['test']:>8}")


if __name__ == "__main__":
    main()
