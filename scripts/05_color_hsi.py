"""
Script 05: Apply homomorphic filtering to color images in HSI space.
Converts an RGB image to HSI, processes only the intensity channel with
homomorphic filtering, and then reconstructs the color image.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.filters import make_butterworth_homomorphic, make_gaussian_homomorphic


if __name__ == "__main__":
    image_path = "images/photo_1.jpg"
    print(f"Loading color image: {image_path}")
    img = Image.open(image_path).convert("RGB")
    rgb = np.array(img, dtype=np.float64)
    print(f"  Shape: {rgb.shape}, range: [{rgb.min()}, {rgb.max()}]")

    gamma_l = 0.5
    gamma_h = 1.8
    d0 = 30
    filter_type = "gaussian"
    print("HSI homomorphic filtering setup:")
    print(f"  Filter type: {filter_type}")
    print(f"  gamma_L={gamma_l}, gamma_H={gamma_h}, D0={d0}")

    print("Converting RGB image to HSI...")
    # TODO: compute H, S, and I channels explicitly
    # TODO: print min and max values for each channel

    print("Applying homomorphic filtering to intensity channel...")
    # TODO: apply the grayscale homomorphic pipeline to the intensity channel only

    print("Reconstructing RGB image from filtered HSI channels...")
    # TODO: convert filtered HSI image back to RGB
    # TODO: clip output values to valid display range

    print("Saving color comparison figures...")
    os.makedirs("results", exist_ok=True)
    # TODO: save original image, intensity channel, and processed color result

    print("Done! Color processing results saved.")
