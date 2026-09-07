# Cucumber Disease Detector — Prototype Pipeline

Prototype for the Chanakya Fellowship crop disease detection project, built
against Cucumber as the first pilot crop. This is a stand-in pipeline using
a public dataset — swap in TIH's own labeled data later with no code
changes (same folder layout).

## 1. Get the dataset

Recommended dataset (real field images, 8 classes, matches cucumber disease
categories well): **Cucumber Disease Recognition Dataset**, Mendeley Data,
Jahangirnagar University —
https://data.mendeley.com/datasets/y6d3z6f8z9/1

Classes: Anthracnose, Bacterial Wilt, Belly Rot, Downy Mildew, Pythium
Fruit Rot, Gummy Stem Blight, Fresh leaves, Fresh cucumber.

Download and unzip it locally (Mendeley isn't reachable from this
sandbox, so grab it on your own machine), you should get one folder per
class.

Alternative / supplementary: the Kaggle "Cucumber Leaf Disease Dataset"
(Powdery mildew & Downy mildew) — smaller but useful for a quick first
sanity-check run: https://www.kaggle.com/datasets/kaushigihanml/cucumber-leaf-disease-dataset

## 2. Set up the environment

```bash
python -m venv venv
source venv/bin/activate     # (or venv\Scripts\activate on Windows)
pip install -r requirements.txt
```

GPU strongly recommended for resnet50/efficientnet/vit/hybrid — CPU-only
will work but training will be slow.

## 3. Split into train/val/test

```bash
python src/split_dataset.py --source /path/to/unzipped_download --dest data --train 0.7 --val 0.15 --test 0.15
```

This produces `data/train/<class>/`, `data/val/<class>/`, `data/test/<class>/`.

## 4. Train each model tier

```bash
python src/train.py --model mobilenet     --data-dir data --epochs 15
python src/train.py --model resnet50      --data-dir data --epochs 15
python src/train.py --model efficientnet  --data-dir data --epochs 15
python src/train.py --model vit           --data-dir data --epochs 15
# Optional, heaviest — only if the above plateau:
python src/train.py --model hybrid        --data-dir data --epochs 15
```

Add `--use-class-weights` if "Fresh leaves"/"Fresh cucumber" dominate the
dataset (likely, since healthy samples are usually easier to collect than
each individual disease).

Each run writes `outputs/<model>/best.pt` (checkpoint) and
`outputs/<model>/history.json` (per-epoch loss/accuracy).

## 5. Evaluate on the held-out test set

```bash
python src/evaluate.py --model resnet50 --checkpoint outputs/resnet50/best.pt --data-dir data
```

Repeat per model. Writes per-class precision/recall/F1 and a confusion
matrix to `outputs/<model>/test_report.json` — this is what goes into the
per-crop technical report deliverable.

## 6. Build the comparison table

```bash
python src/compare_models.py --output-dir outputs
```

Prints and saves a model × metric leaderboard
(`outputs/comparison_table.json`) — this is the "comparative evaluation of
multiple modelling approaches" deliverable for this crop.

## Project structure

```
src/
  dataset.py         # dataloaders + augmentation
  models.py           # model factory (mobilenet/resnet50/efficientnet/vit/hybrid)
  split_dataset.py     # raw download -> train/val/test
  train.py            # train one model tier
  evaluate.py          # test-set metrics + confusion matrix
  compare_models.py    # leaderboard across trained tiers
data/                 # train/val/test images (gitignored — don't commit images)
outputs/              # checkpoints, histories, reports (gitignored)
requirements.txt
```

## Notes / next steps

- **This is a prototype**, not final: once TIH's baseline dataset arrives,
  point `--data-dir` at it (same folder layout) and re-run steps 4–6.
  Expect the public-dataset numbers here to be optimistic vs. real field
  data — treat them as a pipeline sanity check, not a performance claim.
- **Class imbalance**: check `split_dataset.py`'s printed counts before
  training — if any disease class has very few images, consider
  oversampling or heavier augmentation for that class specifically.
- **Mobile export**: once you've picked a winning tier, convert to
  TFLite/ONNX for the deployment pipeline (not included here yet — say
  the word if you want that added next).
- **IP note**: per the fellowship page, IP stays with TIH-IoT/IIT Bombay
  and publication may not be allowed — keep that in mind before pushing
  this repo anywhere public.
