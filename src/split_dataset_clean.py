import argparse
import hashlib
import random
import shutil
from pathlib import Path


def slugify(name: str) -> str:
    return name.strip().replace(" ", "_").replace("-", "_")


def file_hash(path: Path) -> str:
    """Return MD5 hash of an image file."""
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args():
    p = argparse.ArgumentParser(
        description="Create a leakage-safe train/val/test split."
    )
    p.add_argument("--source", required=True)
    p.add_argument("--dest", default="data_clean")
    p.add_argument("--train", type=float, default=0.7)
    p.add_argument("--val", type=float, default=0.15)
    p.add_argument("--test", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    assert abs(args.train + args.val + args.test - 1.0) < 1e-6

    random.seed(args.seed)

    source = Path(args.source)
    dest = Path(args.dest)

    class_dirs = sorted([d for d in source.iterdir() if d.is_dir()])

    if not class_dirs:
        raise SystemExit(f"No class subfolders found in {source}")

    summary = {}
    total_original = 0
    total_unique = 0
    total_duplicates = 0

    # Store hashes globally so identical images cannot cross splits.
    global_hashes = {}

    for class_dir in class_dirs:
        class_name = slugify(class_dir.name)

        images = sorted(
            [
                f
                for f in class_dir.iterdir()
                if f.suffix.lower() in (".jpg", ".jpeg", ".png")
            ]
        )

        total_original += len(images)

        # Remove exact duplicates.
        unique_images = []
        seen_hashes = set()

        for image in images:
            h = file_hash(image)

            # Check if this exact image appeared in another class.
            if h in global_hashes:
                previous_class, previous_file = global_hashes[h]

                if previous_class != class_name:
                    print(
                        "\nWARNING: Identical image found in different classes:"
                    )
                    print(f"  {previous_class}: {previous_file}")
                    print(f"  {class_name}: {image}")
                    print("Please inspect this manually.\n")

            global_hashes[h] = (class_name, image)

            if h not in seen_hashes:
                seen_hashes.add(h)
                unique_images.append(image)

        duplicates_removed = len(images) - len(unique_images)

        total_duplicates += duplicates_removed
        total_unique += len(unique_images)

        # Randomize ONLY after duplicate removal.
        random.shuffle(unique_images)

        n = len(unique_images)

        n_train = int(n * args.train)
        n_val = int(n * args.val)

        splits = {
            "train": unique_images[:n_train],
            "val": unique_images[n_train:n_train + n_val],
            "test": unique_images[n_train + n_val:],
        }

        for split_name, files in splits.items():
            split_class_dir = dest / split_name / class_name
            split_class_dir.mkdir(parents=True, exist_ok=True)

            for f in files:
                target = split_class_dir / f.name

                if target.exists():
                    target.unlink()

                shutil.copy2(f, target)

        summary[class_name] = {
            "original": len(images),
            "unique": len(unique_images),
            "duplicates_removed": duplicates_removed,
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            "test": len(splits["test"]),
        }

    print("\nClean split complete!\n")
    print(
        f"{'class':25}"
        f"{'original':>10}"
        f"{'unique':>10}"
        f"{'removed':>10}"
        f"{'train':>10}"
        f"{'val':>10}"
        f"{'test':>10}"
    )

    for cname, counts in summary.items():
        print(
            f"{cname:25}"
            f"{counts['original']:>10}"
            f"{counts['unique']:>10}"
            f"{counts['duplicates_removed']:>10}"
            f"{counts['train']:>10}"
            f"{counts['val']:>10}"
            f"{counts['test']:>10}"
        )

    print("\nTOTAL")
    print(f"Original images: {total_original}")
    print(f"Unique images:   {total_unique}")
    print(f"Duplicates removed: {total_duplicates}")
    print(f"Output: {dest.resolve()}")


if __name__ == "__main__":
    main()