import time
from pathlib import Path

import torch
from torchvision import datasets

from models import build_model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = Path("data_clean/test")

MODELS = {
    "ViT-B/16": {
        "name": "vit",
        "checkpoint": "outputs/vit/best.pt",
    },
    "ResNet-50": {
        "name": "resnet50",
        "checkpoint": "outputs/resnet50/best.pt",
    },
    "MobileNetV3-Large": {
        "name": "mobilenet",
        "checkpoint": "outputs/mobilenet/best.pt",
    },
    "EfficientNet-B4": {
        "name": "efficientnet",
        "checkpoint": "outputs/efficientnet/best.pt",
    },
}


def load_model(model_name, checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )

    class_names = checkpoint["class_names"]

    model = build_model(
        model_name,
        num_classes=len(class_names),
        pretrained=False,
    )

    model.load_state_dict(checkpoint["model_state"])
    model = model.to(DEVICE)
    model.eval()

    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def get_checkpoint_size(path):
    size_mb = Path(path).stat().st_size / (1024 * 1024)
    return size_mb


def benchmark(model, image_tensor, iterations=100):
    image_tensor = image_tensor.to(DEVICE)

    # Warm-up GPU
    with torch.no_grad():
        for _ in range(20):
            model(image_tensor)

    torch.cuda.synchronize()

    # Reset peak memory measurement
    torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()

    with torch.no_grad():
        for _ in range(iterations):
            model(image_tensor)

    torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    avg_ms = (elapsed / iterations) * 1000
    fps = iterations / elapsed

    peak_memory_mb = (
        torch.cuda.max_memory_allocated() / (1024 * 1024)
    )

    return avg_ms, fps, peak_memory_mb


def main():

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available. This benchmark requires an NVIDIA GPU."
        )

    print("=" * 75)
    print("MODEL BENCHMARK")
    print("=" * 75)

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA:   {torch.version.cuda}")
    print()

    # Load one test image.
    dataset = datasets.ImageFolder(
        DATA_DIR,
        transform=None,
    )

    image, label = dataset[0]

    # Convert PIL image to the same 224x224 ImageNet input
    # used by the project.
    from torchvision import transforms

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])

    image_tensor = preprocess(image).unsqueeze(0)

    results = []

    for display_name, info in MODELS.items():

        print(f"Benchmarking {display_name}...")

        checkpoint_path = info["checkpoint"]

        if not Path(checkpoint_path).exists():
            print(f"  WARNING: {checkpoint_path} not found")
            continue

        model = load_model(
            info["name"],
            checkpoint_path,
        )

        params = count_parameters(model)
        params_m = params / 1_000_000

        checkpoint_mb = get_checkpoint_size(
            checkpoint_path
        )

        avg_ms, fps, memory_mb = benchmark(
            model,
            image_tensor,
            iterations=100,
        )

        results.append({
            "model": display_name,
            "parameters_m": params_m,
            "checkpoint_mb": checkpoint_mb,
            "latency_ms": avg_ms,
            "fps": fps,
            "gpu_memory_mb": memory_mb,
        })

        print(
            f"  Parameters:    {params_m:.2f} M"
        )
        print(
            f"  Checkpoint:    {checkpoint_mb:.2f} MB"
        )
        print(
            f"  Latency:       {avg_ms:.2f} ms/image"
        )
        print(
            f"  Throughput:    {fps:.2f} images/sec"
        )
        print(
            f"  GPU memory:    {memory_mb:.2f} MB"
        )
        print()

        # Free model before loading the next one.
        del model
        torch.cuda.empty_cache()

    print("=" * 75)
    print("FINAL COMPARISON")
    print("=" * 75)

    print(
        f"{'Model':22}"
        f"{'Params(M)':>12}"
        f"{'Size(MB)':>12}"
        f"{'Latency(ms)':>15}"
        f"{'Images/s':>12}"
        f"{'VRAM(MB)':>12}"
    )

    for r in results:
        print(
            f"{r['model']:22}"
            f"{r['parameters_m']:>12.2f}"
            f"{r['checkpoint_mb']:>12.2f}"
            f"{r['latency_ms']:>15.2f}"
            f"{r['fps']:>12.2f}"
            f"{r['gpu_memory_mb']:>12.2f}"
        )

    print()
    print("Benchmark complete.")


if __name__ == "__main__":
    main()