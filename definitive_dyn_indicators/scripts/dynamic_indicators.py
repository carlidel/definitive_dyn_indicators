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


def smallest_alignment_index_6d(
    v1_x, v1_px, v1_y, v1_py, v1_zeta, v1_delta,
    v2_x, v2_px, v2_y, v2_py, v2_zeta, v2_delta):
    sali = np.min([
        np.sqrt(
            (v1_x + v2_x)**2 +
            (v1_px + v2_px)**2 +
            (v1_y + v2_y)**2 +
            (v1_py + v2_py)**2 +
            (v1_zeta + v2_zeta)**2 +
            (v1_delta + v2_delta)**2),
        np.sqrt(
            (v1_x - v2_x)**2 +
            (v1_px - v2_px)**2 +
            (v1_y - v2_y)**2 +
            (v1_py - v2_py)**2 +
            (v1_zeta - v2_zeta)**2 +
            (v1_delta - v2_delta)**2)
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


def global_alignment_index_6d(
    v1_x, v1_px, v1_y, v1_py, v1_zeta, v1_delta,
    v2_x, v2_px, v2_y, v2_py, v2_zeta, v2_delta,
    v3_x, v3_px, v3_y, v3_py, v3_zeta, v3_delta,
    v4_x, v4_px, v4_y, v4_py, v4_zeta, v4_delta,
    v5_x, v5_px, v5_y, v5_py, v5_zeta, v5_delta,
    v6_x, v6_px, v6_y, v6_py, v6_zeta, v6_delta):
    """
    Global Alignment Index 6D
    """
    matrix = np.array([
        [v1_x, v2_x, v3_x, v4_x, v5_x, v6_x],
        [v1_px, v2_px, v3_px, v4_px, v5_px, v6_px],
        [v1_y, v2_y, v3_y, v4_y, v5_y, v6_y],
        [v1_py, v2_py, v3_py, v4_py, v5_py, v6_py],
        [v1_zeta, v2_zeta, v3_zeta, v4_zeta, v5_zeta, v6_zeta],
        [v1_delta, v2_delta, v3_delta, v4_delta, v5_delta, v6_delta]
    ])
    matrix = np.swapaxes(matrix, 1, 2)
    matrix = np.swapaxes(matrix, 0, 1)

    bool_mask = np.all(np.logical_not(np.isnan(matrix)), axis=(1, 2))
    _, s, _ = np.linalg.svd(matrix[bool_mask], full_matrices=True)
    result = np.zeros((len(v1_x)))
    result[np.logical_not(bool_mask)] = np.nan
    result[bool_mask] = np.prod(s, axis=-1)
    return result


def global_alignment_index_4_6d(
    v1_x, v1_px, v1_y, v1_py, v1_zeta, v1_delta,
    v2_x, v2_px, v2_y, v2_py, v2_zeta, v2_delta,
    v3_x, v3_px, v3_y, v3_py, v3_zeta, v3_delta,
    v4_x, v4_px, v4_y, v4_py, v4_zeta, v4_delta):
    """
    Global Alignment Index 4 in 6D version
    """
    matrix = np.array([
        [v1_x, v2_x, v3_x, v4_x],
        [v1_px, v2_px, v3_px, v4_px],
        [v1_y, v2_y, v3_y, v4_y],
        [v1_py, v2_py, v3_py, v4_py],
        [v1_zeta, v2_zeta, v3_zeta, v4_zeta],
        [v1_delta, v2_delta, v3_delta, v4_delta]
    ])
    matrix = np.swapaxes(matrix, 1, 2)
    matrix = np.swapaxes(matrix, 0, 1)

    bool_mask = np.all(np.logical_not(np.isnan(matrix)), axis=(1, 2))
    _, s, _ = np.linalg.svd(matrix[bool_mask], full_matrices=True)
    result = np.zeros((len(v1_x)))
    result[np.logical_not(bool_mask)] = np.nan
    result[bool_mask] = np.prod(s, axis=-1)
    return result