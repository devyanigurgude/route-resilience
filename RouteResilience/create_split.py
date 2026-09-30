import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

IMAGE_DIR = "./images_png"
MASK_DIR = "./masks_png"

files = sorted(
    f for f in os.listdir(IMAGE_DIR)
    if f.endswith(".png")
)

records = []

for filename in files:
    mask_path = os.path.join(MASK_DIR, filename)

    mask = np.array(Image.open(mask_path))

    road_percentage = (mask == 255).mean() * 100

    records.append({
        "filename": filename,
        "road_percentage": road_percentage
    })

df = pd.DataFrame(records)

# Create bins based on road coverage
df["coverage_bin"] = pd.qcut(
    df["road_percentage"],
    q=10,
    labels=False,
    duplicates="drop"
)

# 70% train, 30% temporary
train, temp = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["coverage_bin"]
)

# Split remaining 30% into 15% validation + 15% test
val, test = train_test_split(
    temp,
    test_size=0.50,
    random_state=42,
    stratify=temp["coverage_bin"]
)

train["split"] = "train"
val["split"] = "validation"
test["split"] = "test"

final_df = pd.concat([train, val, test])

final_df = final_df[
    ["filename", "road_percentage", "split"]
].sort_values("filename")

final_df.to_csv("data_split.csv", index=False)

print("Dataset split completed!")
print()
print(final_df["split"].value_counts())
print()
print("Total samples:", len(final_df))
print()
print("Saved as: data_split.csv")