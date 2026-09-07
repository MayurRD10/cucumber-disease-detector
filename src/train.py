"""
Train a single model tier on the cucumber dataset.

Usage:
    python src/train.py --model resnet50 --data-dir data --epochs 15

Run once per model tier (mobilenet, resnet50, efficientnet, vit, hybrid)
to produce the comparison table called for in the project plan. Each run
writes a checkpoint + a JSON history file to outputs/<model>/.
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import build_dataloaders, class_weights
from models import MODEL_REGISTRY, build_model


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--model",
        choices=list(MODEL_REGISTRY.keys()),
        required=True
    )

    p.add_argument(
        "--data-dir",
        default="data"
    )

    p.add_argument(
        "--output-dir",
        default="outputs"
    )

    p.add_argument(
        "--epochs",
        type=int,
        default=15
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=32
    )

    p.add_argument(
        "--image-size",
        type=int,
        default=224
    )

    p.add_argument(
        "--lr",
        type=float,
        default=3e-4
    )

    p.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4
    )

    p.add_argument(
        "--freeze-backbone-epochs",
        type=int,
        default=2,
        help=(
            "Train only the new head for this many epochs before "
            "unfreezing the whole backbone (helps small datasets)."
        )
    )

    p.add_argument(
        "--use-class-weights",
        action="store_true",
        help=(
            "Weight the loss by inverse class frequency. "
            "Not needed for the balanced clean cucumber dataset."
        )
    )

    p.add_argument(
        "--num-workers",
        type=int,
        default=4
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42
    )

    return p.parse_args()


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """
    Freeze/unfreeze everything except the final classification layer(s).

    Works across the different head naming conventions used by torchvision
    models:
        fc         -> ResNet
        classifier -> MobileNet/EfficientNet
        heads      -> ViT

    When trainable=False, the backbone is frozen and only the
    classification head remains trainable.
    """

    # Set every parameter to the requested state.
    for param in model.parameters():
        param.requires_grad = trainable

    # Always keep the classification head trainable.
    head_attr_names = ("fc", "classifier", "heads")

    for attr in head_attr_names:
        head = getattr(model, attr, None)

        if head is not None:
            for param in head.parameters():
                param.requires_grad = True


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train: bool,
) -> tuple[float, float]:
    """
    Run one training or evaluation epoch.

    Returns:
        average_loss, accuracy
    """

    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(train):

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy


def main():

    args = parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # --------------------------------------------------
    # Device
    # --------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(
        f"Device: {device} | "
        f"Model: {MODEL_REGISTRY[args.model]}"
    )

    # --------------------------------------------------
    # Data
    # --------------------------------------------------

    train_loader, val_loader, _, class_names = build_dataloaders(
        args.data_dir,
        args.image_size,
        args.batch_size,
        args.num_workers
    )

    print(
        f"Classes ({len(class_names)}): {class_names}"
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = build_model(
        args.model,
        num_classes=len(class_names)
    ).to(device)

    # --------------------------------------------------
    # Loss function
    # --------------------------------------------------

    # Label smoothing helps reduce overconfidence and can improve
    # generalization on small image datasets.
    if args.use_class_weights:

        weights = class_weights(args.data_dir).to(device)

        criterion = nn.CrossEntropyLoss(
            weight=weights,
            label_smoothing=0.1
        )

    else:

        criterion = nn.CrossEntropyLoss(
            label_smoothing=0.1
        )

    # --------------------------------------------------
    # Optimizer / Scheduler builders
    # --------------------------------------------------

    def _build_optimizer(
        params,
        lr: float
    ) -> AdamW:

        return AdamW(
            params,
            lr=lr,
            weight_decay=args.weight_decay
        )

    def _build_scheduler(
        optim: AdamW,
        t_max: int
    ) -> CosineAnnealingLR:

        return CosineAnnealingLR(
            optim,
            T_max=t_max
        )

    optimizer: AdamW | None = None
    scheduler: CosineAnnealingLR | None = None

    # --------------------------------------------------
    # Output directory
    # --------------------------------------------------

    out_dir = (
        Path(args.output_dir)
        / args.model
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------
    # Training state
    # --------------------------------------------------

    history = []

    best_val_acc = 0.0

    start = time.time()

    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------

    for epoch in range(
        1,
        args.epochs + 1
    ):

        # ==================================================
        # Phase 1:
        # Freeze backbone and train classification head
        # ==================================================

        if (
            epoch == 1
            and args.freeze_backbone_epochs > 0
        ):

            set_backbone_trainable(
                model,
                trainable=False
            )

            trainable_params = [
                p
                for p in model.parameters()
                if p.requires_grad
            ]

            optimizer = _build_optimizer(
                trainable_params,
                lr=args.lr
            )

            scheduler = _build_scheduler(
                optimizer,
                t_max=args.freeze_backbone_epochs
            )

            print(
                f"Epoch {epoch}: "
                "backbone frozen, training head only"
            )

        # ==================================================
        # Phase 1 without freezing
        # ==================================================

        elif (
            epoch == 1
            and args.freeze_backbone_epochs == 0
        ):

            optimizer = _build_optimizer(
                model.parameters(),
                lr=args.lr
            )

            scheduler = _build_scheduler(
                optimizer,
                t_max=args.epochs
            )

            print(
                f"Epoch {epoch}: "
                "training all parameters end-to-end"
            )

        # ==================================================
        # Phase 2:
        # Unfreeze backbone and fine-tune
        # ==================================================

        elif (
            epoch
            == args.freeze_backbone_epochs + 1
        ):

            set_backbone_trainable(
                model,
                trainable=True
            )

            remaining_epochs = (
                args.epochs
                - epoch
                + 1
            )

            optimizer = _build_optimizer(
                model.parameters(),
                lr=args.lr / 3
            )

            scheduler = _build_scheduler(
                optimizer,
                t_max=remaining_epochs
            )

            print(
                f"Epoch {epoch}: "
                "backbone unfrozen, "
                "fine-tuning end-to-end"
            )

        # ==================================================
        # Training
        # ==================================================

        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            train=True,
        )

        # ==================================================
        # Validation
        # ==================================================

        val_loss, val_acc = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer,
            device,
            train=False,
        )

        # Update learning rate
        scheduler.step()

        # ==================================================
        # Logging
        # ==================================================

        elapsed = time.time() - start

        print(
            f"[{elapsed:6.1f}s] "
            f"epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f}"
        )

        # ==================================================
        # Save history
        # ==================================================

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        # ==================================================
        # Save best checkpoint
        # ==================================================

        if val_acc > best_val_acc:

            best_val_acc = val_acc

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "class_names": class_names,
                    "args": vars(args),
                },
                out_dir / "best.pt"
            )

            print(
                f"  ✓ New best model saved "
                f"(val_acc={best_val_acc:.4f})"
            )

    # --------------------------------------------------
    # Save training history
    # --------------------------------------------------

    with open(
        out_dir / "history.json",
        "w"
    ) as f:

        json.dump(
            {
                "model": args.model,
                "class_names": class_names,
                "history": history,
                "best_val_acc": best_val_acc,
            },
            f,
            indent=2
        )

    # --------------------------------------------------
    # Finished
    # --------------------------------------------------

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(
        f"Best validation accuracy: "
        f"{best_val_acc:.4%}"
    )

    print(
        f"Checkpoint: "
        f"{out_dir / 'best.pt'}"
    )

    print(
        f"History: "
        f"{out_dir / 'history.json'}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()