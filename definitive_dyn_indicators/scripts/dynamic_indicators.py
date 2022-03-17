import numpy as np
import pandas as pd
from numba import njit, prange
from tqdm.auto import tqdm


def raw_displacement(x, px, y, py, x_d, px_d, y_d, py_d):
    """
    Lyapunov
    """
    displacement = np.sqrt(
        (x_d - x)**2 + (px_d - px)** 2 + (y_d - y)**2 + (py_d - py)**2)

    lyap = np.log(displacement)
    # add lyap to results
    return lyap


def fast_lyapunov_indicator(x, px, y, py, x_d, px_d, y_d, py_d, mod_d, t):
    """
    Fast Lyapunov Indicator
    """
    displacement = np.sqrt(
        (x_d - x)**2 + (px_d - px)** 2 + (y_d - y)**2 + (py_d - py)**2)

    lyap = np.log(displacement / mod_d) / t
    # add lyap to results
    return lyap


def invariant_lyapunov_error(x, px, y, py, x_1, px_1, y_1, py_1, x_2, px_2, y_2, py_2, x_3, px_3, y_3, py_3, x_4, px_4, y_4, py_4, mod_d, t):
    matrix = np.array([
        [x_1 - x, px_1 - px, y_1 - y, py_1 - py],
        [x_2 - x, px_2 - px, y_2 - y, py_2 - py],
        [x_3 - x, px_3 - px, y_3 - y, py_3 - py],
        [x_4 - x, px_4 - px, y_4 - y, py_4 - py]
    ])
    matrix_transpose = np.transpose(matrix, axes=(2, 1, 0))
    matrix = np.transpose(matrix, axes=(2, 0, 1))
    mult = np.matmul(matrix_transpose, matrix)
    trace = np.trace(mult, axis1=1, axis2=2)
    return np.log(np.sqrt(trace) / mod_d) / t


def orthonormal_lyapunov_indicator(d1, d2, d3, d4, mod_d, t):
    return np.log10(np.sqrt(d1**2 + d2**2 + d3**2 + d4**2) / mod_d) / t


def reversibility_error(x, px, y, py, x_rev, px_rev, y_rev, py_rev):
    """
    Reversibility Error
    """
    displacement = np.sqrt((x_rev - x)**2 + (px_rev - px)
                        ** 2 + (y_rev - y)**2 + (py_rev - py)**2)
    return displacement


def smallest_alignment_index(x_diff_1, px_diff_1, y_diff_1, py_diff_1, x_diff_2, px_diff_2, y_diff_2, py_diff_2):
    """
    Smallest Alignment Index
    """
    sali = np.min([
        np.sqrt(
            (x_diff_1 + x_diff_2)**2 +
            (px_diff_1 + px_diff_2)**2 + 
            (y_diff_1 + y_diff_2)**2 + 
            (py_diff_1 + py_diff_2)**2),
        np.sqrt(
            (x_diff_1 - x_diff_2)**2 + 
            (px_diff_1 - px_diff_2)**2 + 
            (y_diff_1 - y_diff_2)**2 + 
            (py_diff_1 - py_diff_2)**2)
    ], axis=0)

    return sali


def global_alignment_index(x_diff_1, px_diff_1, y_diff_1, py_diff_1, x_diff_2, px_diff_2, y_diff_2, py_diff_2, x_diff_3, px_diff_3, y_diff_3, py_diff_3, x_diff_4, px_diff_4, y_diff_4, py_diff_4):
    """
    Global Alignment Index
    """
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
    result = np.zeros((len(x_diff_1)))
    result[np.logical_not(bool_mask)] = np.nan
    result[bool_mask] = np.prod(s, axis=-1)
    return result
