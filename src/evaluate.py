"""
Evaluate a trained checkpoint on the held-out test set: per-class
precision/recall/F1, confusion matrix, and overall accuracy — the metrics
called for in the project plan's evaluation framework.

Usage:
    python src/evaluate.py --model resnet50 --checkpoint outputs/resnet50/best.pt --data-dir data
"""

import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix

from dataset import build_dataloaders
from models import build_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--output-dir", default="outputs")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader, class_names = build_dataloaders(
        args.data_dir, args.image_size, args.batch_size
    )

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_classes = list(checkpoint.get("class_names", class_names))
    assert ckpt_classes == list(class_names), (
        "Checkpoint's class order doesn't match the current test set's class order — "
        "re-check your data/ folder hasn't changed since training."
    )

    model = build_model(args.model, num_classes=len(class_names), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    report = classification_report(
        all_labels, all_preds, target_names=class_names, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds).tolist()

    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))
    print("Confusion matrix (rows=true, cols=predicted):")
    print(f"{'':>20}" + "".join(f"{c[:8]:>10}" for c in class_names))
    for cname, row in zip(class_names, cm):
        print(f"{cname[:20]:>20}" + "".join(f"{v:>10}" for v in row))

    out_dir = Path(args.output_dir) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "test_report.json", "w") as f:
        json.dump({"classification_report": report, "confusion_matrix": cm,
                    "class_names": class_names}, f, indent=2)

    print(f"\nFull report saved to {out_dir / 'test_report.json'}")


if __name__ == "__main__":
    main()
