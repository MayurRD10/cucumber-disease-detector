import argparse
from pathlib import Path

import torch
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.metrics import accuracy_score
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models import build_model


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def make_transform(mode):
    ops = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]

    if mode == "dark":
        ops.append(transforms.Lambda(
            lambda img: ImageEnhance.Brightness(img).enhance(0.6)
        ))

    elif mode == "bright":
        ops.append(transforms.Lambda(
            lambda img: ImageEnhance.Brightness(img).enhance(1.4)
        ))

    elif mode == "blur":
        ops.append(transforms.Lambda(
            lambda img: img.filter(ImageFilter.GaussianBlur(radius=2))
        ))

    elif mode == "noise":
        ops.append(transforms.ToTensor())
        ops.append(transforms.Lambda(
            lambda x: torch.clamp(x + torch.randn_like(x) * 0.08, 0, 1)
        ))
        ops.append(transforms.Normalize(MEAN, STD))
        return transforms.Compose(ops)

    elif mode == "jpeg":
        ops.append(transforms.Lambda(
            lambda img: jpeg_compress(img, quality=30)
        ))

    ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    return transforms.Compose(ops)


def jpeg_compress(img, quality=30):
    import io

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def evaluate(model, loader, device):
    predictions = []
    labels = []

    with torch.no_grad():
        for images, y in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu()

            predictions.extend(preds.tolist())
            labels.extend(y.tolist())

    return accuracy_score(labels, predictions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data_clean")
    parser.add_argument("--checkpoint", default="outputs/vit/best.pt")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use the clean test set only.
    test_dir = Path(args.data_dir) / "test"

    modes = [
        ("clean", "Clean"),
        ("dark", "Dark / low brightness"),
        ("bright", "Bright / overexposed"),
        ("blur", "Gaussian blur"),
        ("noise", "Gaussian noise"),
        ("jpeg", "JPEG compression"),
    ]

    # Load checkpoint to get the correct class order.
    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )

    class_names = checkpoint["class_names"]

    model = build_model(
        "vit",
        num_classes=len(class_names),
        pretrained=False,
    ).to(device)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    print(f"Device: {device}")
    print("Model: ViT-B/16")
    print(f"Test images: {sum(1 for _ in test_dir.rglob('*') if _.suffix.lower() in {'.jpg', '.jpeg', '.png'})}")
    print()

    results = []

    for mode, description in modes:
        transform = make_transform(mode)

        dataset = datasets.ImageFolder(
            test_dir,
            transform=transform,
        )

        # Make sure ImageFolder class order matches training.
        assert dataset.classes == class_names, (
            f"Class mismatch: {dataset.classes} != {class_names}"
        )

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

        accuracy = evaluate(model, loader, device)

        results.append((description, accuracy))

        print(f"{description:25} {accuracy * 100:6.2f}%")

    print("\nRobustness test complete.")


if __name__ == "__main__":
    main()