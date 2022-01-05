import numpy as np
import pandas as pd
from numba import njit, prange
from tqdm.auto import tqdm

from henon_map_cpp import henon_tracker

def fast_lyapunov_indicator(x, px, y, py, x_d, px_d, y_d, py_d, mod_d, t):
    """
    Fast Lyapunov Indicator
    """
    displacement = np.sqrt(
        (x_d - x)**2 + (px_d - px)** 2 + (y_d - y)**2 + (py_d - py)**2)

    lyap = np.log10(displacement / mod_d) / t
    # add lyap to results
    return lyap


def reversibility_error(x, px, y, py, x_rev, px_rev, y_rev, py_rev):
    """
    Reversibility Error
    """
    displacement = np.sqrt((x_rev - x)**2 + (px_rev - px)
                        ** 2 + (y_rev - y)**2 + (py_rev - py)**2)
    return displacement


def smallest_alignment_index(x, px, y, py, x_d1, px_d1, y_d1, py_d1, x_d2, px_d2, y_d2, py_d2):
    """
    Smallest Alignment Index
    """
    x_diff_1 = x_d1 - x
    px_diff_1 = px_d1 - px
    y_diff_1 = y_d1 - y
    py_diff_1 = py_d1 - py
    norm_1 = np.sqrt(x_diff_1**2 + px_diff_1**2 +
                        y_diff_1**2 + py_diff_1**2)
    x_diff_1 /= norm_1
    px_diff_1 /= norm_1
    y_diff_1 /= norm_1
    py_diff_1 /= norm_1

    x_diff_2 = x_d2 - x
    px_diff_2 = px_d2 - px
    y_diff_2 = y_d2 - y
    py_diff_2 = py_d2 - py
    norm_2 = np.sqrt(x_diff_2**2 + px_diff_2**2 +
                        y_diff_2**2 + py_diff_2**2)
    x_diff_2 /= norm_2
    px_diff_2 /= norm_2
    y_diff_2 /= norm_2
    py_diff_2 /= norm_2

    sali = np.min([
        np.sqrt((x_diff_1 + x_diff_2)**2 + (px_diff_1 + px_diff_2) **
                2 + (y_diff_1 + y_diff_2)**2 + (py_diff_1 + py_diff_2)**2),
        np.sqrt((x_diff_1 - x_diff_2)**2 + (px_diff_1 - px_diff_2) **
                2 + (y_diff_1 - y_diff_2)**2 + (py_diff_1 - py_diff_2)**2)
    ], axis=0)

    return sali


def global_alignment_index(x, px, y, py, x_d1, px_d1, y_d1, py_d1, x_d2, px_d2, y_d2, py_d2, x_d3, px_d3, y_d3, py_d3, x_d4, px_d4, y_d4, py_d4):
    """
    Global Alignment Index
    """
    x_diff_1 = x_d1 - x
    px_diff_1 = px_d1 - px
    y_diff_1 = y_d1 - y
    py_diff_1 = py_d1 - py
    norm_1 = np.sqrt(x_diff_1**2 + px_diff_1**2 +
                        y_diff_1**2 + py_diff_1**2)
    x_diff_1 /= norm_1
    px_diff_1 /= norm_1
    y_diff_1 /= norm_1
    py_diff_1 /= norm_1

    x_diff_2 = x_d2 - x
    px_diff_2 = px_d2 - px
    y_diff_2 = y_d2 - y
    py_diff_2 = py_d2 - py
    norm_2 = np.sqrt(x_diff_2**2 + px_diff_2**2 +
                        y_diff_2**2 + py_diff_2**2)
    x_diff_2 /= norm_2
    px_diff_2 /= norm_2
    y_diff_2 /= norm_2
    py_diff_2 /= norm_2

    x_diff_3 = x_d3 - x
    px_diff_3 = px_d3 - px
    y_diff_3 = y_d3 - y
    py_diff_3 = py_d3 - py
    norm_3 = np.sqrt(x_diff_3**2 + px_diff_3**2 +
                        y_diff_3**2 + py_diff_3**2)
    x_diff_3 /= norm_3
    px_diff_3 /= norm_3
    y_diff_3 /= norm_3
    py_diff_3 /= norm_3

    x_diff_4 = x_d4 - x
    px_diff_4 = px_d4 - px
    y_diff_4 = y_d4 - y
    py_diff_4 = py_d4 - py
    norm_4 = np.sqrt(x_diff_4**2 + px_diff_4**2 +
                        y_diff_4**2 + py_diff_4**2)
    x_diff_4 /= norm_4
    px_diff_4 /= norm_4
    y_diff_4 /= norm_4
    py_diff_4 /= norm_4

    matrix = np.array([
        [x_diff_1, x_diff_2, x_diff_3, x_diff_4],
        [px_diff_1, px_diff_2, px_diff_3, px_diff_4],
        [y_diff_1, y_diff_2, y_diff_3, y_diff_4],
        [py_diff_1, py_diff_2, py_diff_3, py_diff_4]
    ])
    matrix = np.swapaxes(matrix, 1, 2)
    matrix = np.swapaxes(matrix, 0, 1)

    bool_mask = np.all(np.logical_not(np.isnan(matrix)), axis=(1, 2))
    _, s, _ = np.linalg.svd(matrix[bool_mask], full_matrices=True)
    result = np.zeros((len(x)))
    result[np.logical_not(bool_mask)] = np.nan
    result[bool_mask] = np.prod(s, axis=-1)
    return result