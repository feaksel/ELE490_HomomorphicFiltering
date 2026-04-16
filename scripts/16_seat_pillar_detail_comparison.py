"""
Script 16: Detail comparison for seat and pillar.
This script highlights texture/detail changes using the brightened
homomorphic outputs, showing both full-frame context and zoomed crops.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.patches import Rectangle


DETAIL_CASES = [
    {
        "name": "seat",
        "grayscale_path": os.path.join("results", "real_batch", "seat_grayscale.png"),
        "homomorphic_path": os.path.join("results", "real_batch", "seat_homomorphic_standard_bright.png"),
        "crop": (260, 430, 520, 520),
    },
    {
        "name": "pillar",
        "grayscale_path": os.path.join("results", "real_batch", "pillar_grayscale.png"),
        "homomorphic_path": os.path.join("results", "real_batch", "pillar_homomorphic_standard_bright.png"),
        "crop": (560, 200, 520, 520),
    },
]


def load_uint8_image(image_path):
    return np.array(Image.open(image_path).convert("L"), dtype=np.uint8)


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    print("Running seat and pillar detail comparison...")
    figure, axes = plt.subplots(len(DETAIL_CASES), 4, figsize=(16, 7 * len(DETAIL_CASES)))
    if len(DETAIL_CASES) == 1:
        axes = np.array([axes])

    for row_index, case in enumerate(DETAIL_CASES):
        gray = load_uint8_image(case["grayscale_path"])
        homomorphic = load_uint8_image(case["homomorphic_path"])
        crop_x, crop_y, crop_w, crop_h = case["crop"]

        gray_crop = gray[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]
        homomorphic_crop = homomorphic[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]

        print(f"Processed detail case: {case['name']}")
        print(f"  Crop rectangle: x={crop_x}, y={crop_y}, w={crop_w}, h={crop_h}")

        for axis, image, title in [
            (axes[row_index, 0], gray, f"{case['name']}: Grayscale"),
            (axes[row_index, 1], homomorphic, f"{case['name']}: Standard HF + Brightness"),
        ]:
            axis.imshow(image, cmap="gray", vmin=0, vmax=255)
            axis.add_patch(Rectangle((crop_x, crop_y), crop_w, crop_h, linewidth=2, edgecolor="red", facecolor="none"))
            axis.set_title(title)
            axis.set_xlabel("Column")
            axis.set_ylabel("Row")

        axes[row_index, 2].imshow(gray_crop, cmap="gray", vmin=0, vmax=255)
        axes[row_index, 2].set_title(f"{case['name']}: Grayscale detail")
        axes[row_index, 2].set_xlabel("Column")
        axes[row_index, 2].set_ylabel("Row")

        axes[row_index, 3].imshow(homomorphic_crop, cmap="gray", vmin=0, vmax=255)
        axes[row_index, 3].set_title(f"{case['name']}: HF + Bright detail")
        axes[row_index, 3].set_xlabel("Column")
        axes[row_index, 3].set_ylabel("Row")

    figure.suptitle("Seat and Pillar Detail Comparison with Brightened Homomorphic Outputs")
    figure.tight_layout()
    figure.savefig(os.path.join("results", "seat_pillar_detail_comparison.png"), dpi=200)
    plt.close(figure)

    print("Done! Seat and pillar detail comparison saved.")
