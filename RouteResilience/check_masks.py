from PIL import Image
import os
import numpy as np

folder = r".\masks_png"

results = []

for filename in os.listdir(folder):
    if filename.endswith(".png"):
        path = os.path.join(folder, filename)
        mask = np.array(Image.open(path))

        road_percentage = (mask == 255).mean() * 100

        results.append((road_percentage, filename))

results.sort()

print("LOWEST ROAD COVERAGE:")
for percentage, filename in results[:5]:
    print(f"{filename}: {percentage:.4f}%")

print("\nHIGHEST ROAD COVERAGE:")
for percentage, filename in results[-5:]:
    print(f"{filename}: {percentage:.4f}%")