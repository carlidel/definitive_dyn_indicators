import numpy as np
import numpy.random
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
import pickle
import os

displacement_scale = 1e-6
random_seed = 42

# set random seed
rs = RandomState(MT19937(SeedSequence(random_seed)))

def sample_4d_displacement_on_a_sphere():
    n = rs.uniform(-1, 1, size=4)
    while n[0]**2 + n[1]**2 > 1 or n[2]**2 + n[3]**2 > 1:
        n = rs.uniform(-1, 1, size=4)
    fix = (1 - n[0]**2 - n[1]**2) / (n[2]**2 - n[3]**2)
    n[2] *= fix
    n[3] *= fix
    return n 

lhc_config = {
    'name': 'lhc_square_universal_config',
    'samples': 200,
    'random_seed': random_seed,

    'x_extents': [0.0, 2e-3],
    'y_extents': [0.0, 2e-3],

    'tracking': 10000,
    'long_tracking': 10000000,

    't_base_10': np.logspace(1, 5, 29, dtype=int),
    't_base_2': np.logspace(1, 17, 17, dtype=int, base=2),
    't_linspace': np.arange(10000, 130001, 10000),

    'zeta_0': 0.0,
    'zeta_1': 0.15,
    'zeta_2': 0.31,
}

lhc_config['t_list'] = np.concatenate((lhc_config['t_base_10'], lhc_config['t_base_2'], np.arange(10000, 100000, 10000), np.arange(100000, 130001, 10000)[1:]))
lhc_config['t_list'] = np.sort(lhc_config['t_list'])

lhc_config["x_sample"], lhc_config["dx"] = np.linspace(
    lhc_config["x_extents"][0],
    lhc_config["x_extents"][1],
    lhc_config["samples"],
    retstep=True
)

lhc_config["y_sample"], lhc_config["dy"] = np.linspace(
    lhc_config["y_extents"][0],
    lhc_config["y_extents"][1],
    lhc_config["samples"],
    retstep=True
)

xx, yy = np.meshgrid(
    lhc_config["x_sample"],
    lhc_config["y_sample"]
)

lhc_config["x_flat"] = xx.flatten()
lhc_config["y_flat"] = yy.flatten()
lhc_config["px_flat"] = np.zeros_like(xx.flatten())
lhc_config["py_flat"] = np.zeros_like(xx.flatten())

lhc_config["total_samples"] = lhc_config["x_flat"].size

lhc_config["displacement"] = min(lhc_config["dx"], lhc_config["dy"]) * displacement_scale

displacement_table = np.array([sample_4d_displacement_on_a_sphere() for _ in range(lhc_config["total_samples"])])

lhc_config["x_random_displacement"] = lhc_config["x_flat"] + displacement_table[:, 0] * lhc_config["displacement"]
lhc_config["y_random_displacement"] = lhc_config["y_flat"] + displacement_table[:, 1] * lhc_config["displacement"]
lhc_config["px_random_displacement"] = lhc_config["px_flat"] + displacement_table[:, 2] * lhc_config["displacement"]
lhc_config["py_random_displacement"] = lhc_config["py_flat"] + displacement_table[:, 3] * lhc_config["displacement"]

lhc_config["x_x_displacement"] = lhc_config["x_flat"] + lhc_config["displacement"]
lhc_config["y_x_displacement"] = lhc_config["y_flat"]
lhc_config["px_x_displacement"] = lhc_config["px_flat"]
lhc_config["py_x_displacement"] = lhc_config["py_flat"]

lhc_config["x_y_displacement"] = lhc_config["x_flat"]
lhc_config["y_y_displacement"] = lhc_config["y_flat"] + lhc_config["displacement"]
lhc_config["px_y_displacement"] = lhc_config["px_flat"]
lhc_config["py_y_displacement"] = lhc_config["py_flat"]

lhc_config["x_px_displacement"] = lhc_config["x_flat"]
lhc_config["y_px_displacement"] = lhc_config["y_flat"]
lhc_config["px_px_displacement"] = lhc_config["px_flat"] + lhc_config["displacement"]
lhc_config["py_px_displacement"] = lhc_config["py_flat"]

lhc_config["x_py_displacement"] = lhc_config["x_flat"]
lhc_config["y_py_displacement"] = lhc_config["y_flat"]
lhc_config["px_py_displacement"] = lhc_config["px_flat"]
lhc_config["py_py_displacement"] = lhc_config["py_flat"] + lhc_config["displacement"]

lhc_config["selected_masks"] = [
    {"beam": 1, "seed": 33, "quality": "worse"},
    {"beam": 1, "seed": 11, "quality": "average"},
    {"beam": 1, "seed": 21, "quality": "best"},
    {"beam": 2, "seed": 55, "quality": "worse"},
    {"beam": 2, "seed": 18, "quality": "average"},
    {"beam": 2, "seed": 38, "quality": "best"},
]

lhc_config["selected_masks_full"] = [
    "lhc_mask_b1_without_bb_33.json",
    "lhc_mask_b1_without_bb_11.json",
    "lhc_mask_b1_without_bb_21.json",
    "lhc_mask_b4_without_bb_55.json",
    "lhc_mask_b4_without_bb_18.json",
    "lhc_mask_b4_without_bb_38.json",
]

# save lhc_config in the same directory of this script
with open(os.path.join(os.path.dirname(__file__),'global_config.pkl'), 'wb') as f:
    pickle.dump(lhc_config, f)
