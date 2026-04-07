"""
Script 01: Create synthetic unevenly illuminated image.
Takes the cameraman image and multiplies it by a smooth 2D gradient
to simulate non-uniform illumination. Saves both the gradient and
the corrupted image for later use.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


if __name__ == "__main__":
    print("Loading cameraman image...")
    img = Image.open("images/cameraman.tif").convert("L")
    f = np.array(img, dtype=np.float64)
    print(f"  Image shape: {f.shape}, range: [{f.min()}, {f.max()}]")

    print("Creating smooth illumination gradient...")
    # TODO: create a 2D gradient that is bright on one side and dim on the other
    # Hint: use a 2D Gaussian centered off to one side, or a linear ramp
    # The gradient should range roughly from 0.3 to 1.0
    rows, cols = f.shape
    illumination = np.ones((rows, cols), dtype=np.float64)
    # illumination = ...
    print(f"  Illumination range: [{illumination.min():.2f}, {illumination.max():.2f}]")

    print("Applying illumination to create corrupted image...")
    # TODO: multiply the original image by the illumination gradient
    f_corrupted = f.copy()
    # f_corrupted = ...
    f_corrupted = np.clip(f_corrupted, 0, 255)
    print(f"  Corrupted image range: [{f_corrupted.min():.2f}, {f_corrupted.max():.2f}]")

    print("Saving results...")
    os.makedirs("results", exist_ok=True)
    # TODO: save corrupted image to images/synthetic_uneven.png
    # TODO: save a figure showing original, illumination gradient, and corrupted image side by side

    print("Done! Synthetic image saved.")
