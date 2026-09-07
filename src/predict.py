"""
Run a trained checkpoint on a single image (or a folder of images) and
print the predicted class + confidence + full probability breakdown.

This is the first building block toward a demo app: everything a
Streamlit/Flask/CLI frontend needs is already here as a plain function
(`predict_image`) you can import directly.

Usage (single image):
    python src/predict.py --model vit --checkpoint outputs/vit/best.pt --image path/to/leaf.jpg

Usage (folder of images):
    python src/predict.py --model vit --checkpoint outputs/vit/best.pt --image-dir path/to/folder

Usage (as a library, e.g. from a Streamlit app):
    from predict import load_predictor
    predictor = load_predictor("vit", "outputs/vit/best.pt")
    result = predictor(pil_image)  # -> {"prediction": ..., "confidence": ..., "probabilities": {...}}
"""

import argparse
import json
from pathlib import Path
from typing import Callable, Dict

import torch
import torch.nn.functional as F
from PIL import Image

from dataset import build_transforms
from models import build_model

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_predictor(model_name: str, checkpoint_path: str, image_size: int = 224,
                    device: str = None) -> Callable[[Image.Image], Dict]:
    """Loads a checkpoint once and returns a function that predicts on a
    PIL Image. Reuse the returned function across many images/requests —
    don't reload the checkpoint per call (that's what makes this usable
    inside a Streamlit/Flask app without a multi-second lag per request)."""

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    class_names = checkpoint["class_names"]

    model = build_model(model_name, num_classes=len(class_names), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    _, eval_transform = build_transforms(image_size)

    def predict(image: Image.Image) -> Dict:
        image = image.convert("RGB")
        tensor = eval_transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu()

        prob_dict = {cname: round(float(p), 4) for cname, p in zip(class_names, probs)}
        top_idx = int(probs.argmax())

        return {
            "prediction": class_names[top_idx],
            "confidence": round(float(probs[top_idx]), 4),
            "probabilities": dict(sorted(prob_dict.items(), key=lambda kv: kv[1], reverse=True)),
        }

    return predict


def print_result(image_path: Path, result: Dict):
    print(f"\n{image_path.name}")
    print(f"  Prediction: {result['prediction']}  (confidence: {result['confidence']*100:.1f}%)")
    print("  Full breakdown:")
    for cname, prob in result["probabilities"].items():
        bar = "#" * int(prob * 30)
        print(f"    {cname:20} {prob*100:5.1f}%  {bar}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="e.g. vit, resnet50, mobilenet, efficientnet, hybrid")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image", help="Path to a single image")
    p.add_argument("--image-dir", help="Path to a folder of images (processes all of them)")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--output-json", help="Optional path to save results as JSON")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.image and not args.image_dir:
        raise SystemExit("Provide either --image or --image-dir")

    predictor = load_predictor(args.model, args.checkpoint, args.image_size)

    if args.image:
        image_paths = [Path(args.image)]
    else:
        image_paths = sorted(
            f for f in Path(args.image_dir).iterdir()
            if f.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not image_paths:
            raise SystemExit(f"No images found in {args.image_dir}")

    all_results = {}
    for path in image_paths:
        image = Image.open(path)
        result = predictor(image)
        print_result(path, result)
        all_results[str(path)] = result

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved results to {args.output_json}")


if __name__ == "__main__":
    main()
