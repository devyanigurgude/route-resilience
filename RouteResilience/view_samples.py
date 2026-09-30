from PIL import Image
import matplotlib.pyplot as plt

# Sample files
samples = ["101", "4546"]

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

for i, sample in enumerate(samples):

    image_path = f"./images_png/{sample}.png"
    mask_path = f"./masks_png/{sample}.png"

    image = Image.open(image_path)
    mask = Image.open(mask_path)

    # Satellite image
    axes[i, 0].imshow(image)
    axes[i, 0].set_title(f"Satellite Image - {sample}")
    axes[i, 0].axis("off")

    # Road mask
    axes[i, 1].imshow(mask, cmap="gray")
    axes[i, 1].set_title(f"Road Mask - {sample}")
    axes[i, 1].axis("off")

plt.tight_layout()
plt.show()