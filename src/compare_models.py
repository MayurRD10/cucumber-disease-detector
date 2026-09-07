"""
After running train.py + evaluate.py for each model tier, run this to
produce the comparative table the fellowship deliverables call for.

Usage:
    python src/compare_models.py --output-dir outputs
"""

import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="outputs")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)

    rows = []
    for model_dir in sorted(out_dir.iterdir()):
        history_path = model_dir / "history.json"
        report_path = model_dir / "test_report.json"
        if not history_path.exists():
            continue

        history = json.loads(history_path.read_text())
        row = {
            "model": history["model"],
            "best_val_acc": round(history["best_val_acc"], 4),
        }

        if report_path.exists():
            report = json.loads(report_path.read_text())["classification_report"]
            row["test_accuracy"] = round(report["accuracy"], 4)
            row["macro_f1"] = round(report["macro avg"]["f1-score"], 4)
            row["weighted_f1"] = round(report["weighted avg"]["f1-score"], 4)
        else:
            row["test_accuracy"] = row["macro_f1"] = row["weighted_f1"] = "—"

        rows.append(row)

    if not rows:
        print(f"No results found under {out_dir}/. Run train.py (and evaluate.py) first.")
        return

    headers = ["model", "best_val_acc", "test_accuracy", "macro_f1", "weighted_f1"]
    widths = {
        h: max(len(h), *(len(str(r[h])) for r in rows)) + 2
        for h in headers
    }

    def _sort_key(row: dict) -> float:
        """Sort by test accuracy (descending); models without results go last."""
        acc = row["test_accuracy"]
        return acc if isinstance(acc, float) else -1.0

    print("".join(h.ljust(widths[h]) for h in headers))
    print("─" * sum(widths.values()))
    for r in sorted(rows, key=_sort_key, reverse=True):
        print("".join(str(r[h]).ljust(widths[h]) for h in headers))

    with open(out_dir / "comparison_table.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved to {out_dir / 'comparison_table.json'}")


if __name__ == "__main__":
    main()
