import os

image_folder = "./images_png"

files = [f for f in os.listdir(image_folder) if f.endswith(".png")]

print("Total images:", len(files))
print("\nFirst 20 filenames:")

for f in sorted(files)[:20]:
    print(f)