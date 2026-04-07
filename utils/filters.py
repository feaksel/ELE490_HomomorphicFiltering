"""
Homomorphic filter generation functions.
Butterworth and Gaussian variants with gamma_L and gamma_H parameters.
"""
import numpy as np


def make_butterworth_homomorphic(rows, cols, d0, gamma_l, gamma_h, c=1.0, order=2):
    """
    Create a Butterworth homomorphic filter.

    Parameters:
        rows, cols: image dimensions
        d0: cutoff frequency
        gamma_l: low frequency gain (< 1 to compress illumination)
        gamma_h: high frequency gain (> 1 to boost reflectance)
        c: controls sharpness of transition (default 1.0)
        order: filter order (default 2)

    Returns:
        H: 2D filter array of shape (rows, cols), centered
    """
    center_row = rows // 2
    center_col = cols // 2

    u = np.arange(rows) - center_row
    v = np.arange(cols) - center_col
    U, V = np.meshgrid(v, u)
    D = np.sqrt(U ** 2 + V ** 2).astype(np.float64)

    D[D == 0] = 1e-10

    H_hp = 1.0 / (1.0 + (d0 / D) ** (2 * order))
    H = (gamma_h - gamma_l) * H_hp + gamma_l

    return H


def make_gaussian_homomorphic(rows, cols, d0, gamma_l, gamma_h):
    """
    Create a Gaussian homomorphic filter.

    Parameters:
        rows, cols: image dimensions
        d0: cutoff frequency
        gamma_l: low frequency gain (< 1 to compress illumination)
        gamma_h: high frequency gain (> 1 to boost reflectance)

    Returns:
        H: 2D filter array of shape (rows, cols), centered
    """
    center_row = rows // 2
    center_col = cols // 2

    u = np.arange(rows) - center_row
    v = np.arange(cols) - center_col
    U, V = np.meshgrid(v, u)
    D_sq = (U ** 2 + V ** 2).astype(np.float64)

    H_hp = 1.0 - np.exp(-D_sq / (2.0 * d0 ** 2))
    H = (gamma_h - gamma_l) * H_hp + gamma_l

    return H
