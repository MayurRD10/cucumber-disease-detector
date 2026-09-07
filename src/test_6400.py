"""
Evaluate the ViT model on the complete 6,400-image Hugging Face
Cucumber Leaf dataset stored as Parquet files.

IMPORTANT:
This is NOT an independent test set because these images overlap
with the original/augmented dataset used during development.

It is useful for:
- Overall performance analysis
- Per-class accuracy
- Split-wise performance
- Confusion matrix
- Finding systematic errors
"""

import argparse
import json
import glob
import os
from collections import defaultdict

import pandas as pd
import torch
from PIL import Image
from io import BytesIO
from torchvision import transforms

from models import build_model


# --------------------------------------------------
# Arguments
# --------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        default=r"C:\Users\Mayur\Downloads\Cucumber_leaf\data",
        help="Directory containing the Parquet files"
    )

    parser.add_argument(
        "--checkpoint",
        default="outputs/vit/best.pt",
        help="Path to model checkpoint"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32
    )

    parser.add_argument(
        "--output",
        default="outputs/vit/test_6400_report.json"
    )

    return parser.parse_args()


# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# --------------------------------------------------
# Read all Parquet files
# --------------------------------------------------

def load_dataset(data_dir):

    parquet_files = sorted(
        glob.glob(
            os.path.join(data_dir, "*.parquet")
        )
    )

    if not parquet_files:
        raise FileNotFoundError(
            f"No Parquet files found in {data_dir}"
        )

    print("Parquet files:")

    for file in parquet_files:
        print(
            f"  {os.path.basename(file)}"
        )

    datasets = []

    for file in parquet_files:

        df = pd.read_parquet(file)

        filename = os.path.basename(file)

        if filename.startswith("train-"):
            split = "train"

        elif filename.startswith("validation-"):
            split = "validation"

        elif filename.startswith("test-"):
            split = "test"

        else:
            split = "unknown"

        df["split"] = split

        datasets.append(df)

    combined = pd.concat(
        datasets,
        ignore_index=True
    )

    return combined


# --------------------------------------------------
# Evaluate
# --------------------------------------------------

def evaluate(
    df,
    model,
    class_names,
    device,
    batch_size
):

    num_classes = len(class_names)

    # Overall confusion matrix
    confusion = torch.zeros(
        num_classes,
        num_classes,
        dtype=torch.long
    )

    # Split-specific confusion matrices
    split_confusions = defaultdict(
        lambda: torch.zeros(
            num_classes,
            num_classes,
            dtype=torch.long
        )
    )

    model.eval()

    total = 0
    correct = 0

    split_totals = defaultdict(int)
    split_correct = defaultdict(int)

    # --------------------------------------------------
    # Hugging Face → Model label mapping
    # --------------------------------------------------

    # Hugging Face dataset:
    #
    # 0 = Anthracnose
    # 1 = Bacterial Wilt
    # 2 = Belly Rot
    # 3 = Downy Mildew
    # 4 = Gummy Stem Blight
    # 5 = Healthy Cucumber
    # 6 = Healthy Leaf
    # 7 = Pythium Fruit Rot
    #
    # Model:
    #
    # 0 = Anthracnose
    # 1 = Bacterial_Wilt
    # 2 = Belly_Rot
    # 3 = Downy_Mildew
    # 4 = Fresh_Cucumber
    # 5 = Fresh_Leaf
    # 6 = Gummy_Stem_Blight
    # 7 = Pythium_Fruit_Rot

    hf_to_model_label = {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 6,
        5: 4,
        6: 5,
        7: 7,
    }

    # --------------------------------------------------
    # Process images in batches
    # --------------------------------------------------

    for start in range(
        0,
        len(df),
        batch_size
    ):

        batch_df = df.iloc[
            start:start + batch_size
        ]

        images = []
        labels = []
        splits = []

        for _, row in batch_df.iterrows():

            image_bytes = row["image"]["bytes"]

            image = Image.open(
                BytesIO(image_bytes)
            ).convert("RGB")

            image = transform(image)

            images.append(image)

            # Convert Hugging Face label to the
            # model's label numbering.
            hf_label = int(row["label"])

            model_label = hf_to_model_label[
                hf_label
            ]

            labels.append(
                model_label
            )

            splits.append(
                row["split"]
            )

        images = torch.stack(
            images
        ).to(device)

        labels_tensor = torch.tensor(
            labels,
            dtype=torch.long
        ).to(device)

        # --------------------------------------------------
        # Prediction
        # --------------------------------------------------

        with torch.no_grad():

            outputs = model(images)

            predictions = outputs.argmax(
                dim=1
            )

        # --------------------------------------------------
        # Statistics
        # --------------------------------------------------

        for true_label, pred_label, split in zip(
            labels,
            predictions.cpu().tolist(),
            splits
        ):

            confusion[
                true_label,
                pred_label
            ] += 1

            split_confusions[
                split
            ][
                true_label,
                pred_label
            ] += 1

            total += 1

            split_totals[
                split
            ] += 1

            if true_label == pred_label:

                correct += 1

                split_correct[
                    split
                ] += 1

        # --------------------------------------------------
        # Progress
        # --------------------------------------------------

        if (
            start == 0
            or
            (start + batch_size) % 640 == 0
            or
            start + batch_size >= len(df)
        ):

            processed = min(
                start + batch_size,
                len(df)
            )

            print(
                f"Processed "
                f"{processed:,}/{len(df):,} images"
            )

    # --------------------------------------------------
    # Overall accuracy
    # --------------------------------------------------

    overall_accuracy = (
        correct / total
    )

    # --------------------------------------------------
    # Split results
    # --------------------------------------------------

    split_results = {}

    for split in [
        "train",
        "validation",
        "test"
    ]:

        if split in split_totals:

            split_results[split] = {
                "images": split_totals[split],
                "correct": split_correct[split],
                "accuracy": (
                    split_correct[split]
                    / split_totals[split]
                )
            }

    # --------------------------------------------------
    # Per-class results
    # --------------------------------------------------

    per_class = {}

    for i, class_name in enumerate(
        class_names
    ):

        true_count = int(
            confusion[i].sum().item()
        )

        correct_count = int(
            confusion[i, i].item()
        )

        accuracy = (
            correct_count / true_count
            if true_count > 0
            else 0.0
        )

        per_class[class_name] = {
            "images": true_count,
            "correct": correct_count,
            "accuracy": accuracy
        }

    return (
        overall_accuracy,
        split_results,
        per_class,
        confusion
    )


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    args = parse_args()

    # --------------------------------------------------
    # Device
    # --------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nDevice: {device}"
    )

    # --------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------

    print(
        f"Loading checkpoint: "
        f"{args.checkpoint}"
    )

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False
    )

    class_names = checkpoint[
        "class_names"
    ]

    print(
        f"Classes ({len(class_names)}): "
        f"{class_names}"
    )

    # --------------------------------------------------
    # Build model
    # --------------------------------------------------

    model = build_model(
        "vit",
        num_classes=len(class_names)
    )

    model.load_state_dict(
        checkpoint["model_state"]
    )

    model = model.to(device)

    # --------------------------------------------------
    # Load dataset
    # --------------------------------------------------

    print(
        "\nReading Hugging Face dataset..."
    )

    df = load_dataset(
        args.data_dir
    )

    print(
        f"\nTotal images: {len(df):,}"
    )

    print(
        "\nSplit counts:"
    )

    print(
        df["split"].value_counts()
    )

    # --------------------------------------------------
    # Evaluate
    # --------------------------------------------------

    print(
        "\nStarting evaluation..."
    )

    (
        overall_accuracy,
        split_results,
        per_class,
        confusion
    ) = evaluate(
        df,
        model,
        class_names,
        device,
        args.batch_size
    )

    # --------------------------------------------------
    # Print results
    # --------------------------------------------------

    print()
    print("=" * 70)
    print("6,400 IMAGE EVALUATION")
    print("=" * 70)

    print(
        f"\nOverall accuracy: "
        f"{overall_accuracy:.4%}"
    )

    print(
        f"Correct: "
        f"{int(overall_accuracy * len(df)):,}"
        f"/{len(df):,}"
    )

    # --------------------------------------------------
    # Split results
    # --------------------------------------------------

    print(
        "\nSplit results:"
    )

    for split, result in split_results.items():

        print(
            f"  {split.capitalize():12s}: "
            f"{result['accuracy']:.4%} "
            f"({result['correct']}/"
            f"{result['images']})"
        )

    # --------------------------------------------------
    # Per-class results
    # --------------------------------------------------

    print(
        "\nPer-class accuracy:"
    )

    for class_name, result in per_class.items():

        print(
            f"  {class_name:20s}: "
            f"{result['accuracy']:.4%} "
            f"({result['correct']}/"
            f"{result['images']})"
        )

    # --------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------

    print(
        "\nConfusion matrix "
        "(rows=true, cols=predicted):"
    )

    print(
        " " * 22
        + " ".join(
            f"{i:5d}"
            for i in range(
                len(class_names)
            )
        )
    )

    for i, class_name in enumerate(
        class_names
    ):

        values = " ".join(
            f"{int(confusion[i, j]):5d}"
            for j in range(
                len(class_names)
            )
        )

        print(
            f"{class_name:20s} "
            f"{values}"
        )

    # --------------------------------------------------
    # Save JSON report
    # --------------------------------------------------

    report = {
        "warning": (
            "This 6400-image evaluation is not an "
            "independent test because the images overlap "
            "with the original/augmented dataset."
        ),
        "label_mapping": {
            "hugging_face_to_model": {
                "0": 0,
                "1": 1,
                "2": 2,
                "3": 3,
                "4": 6,
                "5": 4,
                "6": 5,
                "7": 7
            }
        },
        "checkpoint": args.checkpoint,
        "total_images": len(df),
        "overall_accuracy": overall_accuracy,
        "split_results": split_results,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
        "class_names": class_names
    }

    output_path = args.output

    output_dir = os.path.dirname(
        output_path
    )

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True
        )

    with open(
        output_path,
        "w"
    ) as f:

        json.dump(
            report,
            f,
            indent=2
        )

    print()
    print(
        f"Full report saved to: "
        f"{output_path}"
    )

    print(
        "\nEvaluation complete."
    )


if __name__ == "__main__":
    main()