import os
import glob
import hashlib
import pandas as pd

HF_DATASET = r"C:\Users\Mayur\Downloads\Cucumber_leaf\data"
ORIGINAL_DATASET = r"C:\Users\Mayur\Downloads\Cucumber Disease Recognition Dataset"


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------
# 1. Hash existing original images
# --------------------------------------------------

print("Hashing existing dataset...")

existing_hashes = {}

for root, _, files in os.walk(ORIGINAL_DATASET):
    for filename in files:
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(root, filename)

            with open(path, "rb") as f:
                image_bytes = f.read()

            h = sha256_bytes(image_bytes)
            existing_hashes[h] = path

print(f"Existing images found: {len(existing_hashes)}")


# --------------------------------------------------
# 2. Read Hugging Face dataset
# --------------------------------------------------

print("\nReading Hugging Face dataset...")

parquet_files = glob.glob(
    os.path.join(HF_DATASET, "*.parquet")
)

hf_total = 0
hf_duplicates = 0
hf_unique = 0

hf_seen = set()
duplicates_against_original = []

for parquet_file in parquet_files:

    print("Processing:", os.path.basename(parquet_file))

    df = pd.read_parquet(parquet_file)

    for _, row in df.iterrows():

        image_bytes = row["image"]["bytes"]

        h = sha256_bytes(image_bytes)

        hf_total += 1

        # Duplicate inside HF dataset
        if h in hf_seen:
            continue

        hf_seen.add(h)
        hf_unique += 1

        # Duplicate against original dataset
        if h in existing_hashes:
            hf_duplicates += 1
            duplicates_against_original.append(
                (h, existing_hashes[h])
            )


# --------------------------------------------------
# 3. Results
# --------------------------------------------------

print("\n" + "=" * 50)
print("DUPLICATE ANALYSIS")
print("=" * 50)

print(f"Hugging Face images:          {hf_total}")
print(f"Unique HF images:             {hf_unique}")
print(f"HF duplicates internally:     {hf_total - hf_unique}")
print(f"Exact matches with original:  {hf_duplicates}")
print(f"Potentially new images:       {hf_unique - hf_duplicates}")

print("=" * 50)

if duplicates_against_original:

    print("\nExamples of duplicates:")

    for h, path in duplicates_against_original[:10]:
        print(path)

else:

    print("\nNo exact duplicates found against original dataset.")