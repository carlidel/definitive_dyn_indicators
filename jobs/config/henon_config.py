import numpy as np
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

henon_config = {
    'name': 'henon_square_universal_config',
    'samples': 1000,
    'random_seed': random_seed,

    'x_extents': [0.0, 0.8],
    'y_extents': [0.0, 0.8],

    'low_tracking': 1000000,
    'tracking': 10000000,
    'extreme_tracking': 100000000,

    't_base_10': np.logspace(1, 6, 51, dtype=int),
    't_base_2': np.logspace(1, 20, 20, dtype=int, base=2),
}

henon_config['t_list'] = np.concatenate(
    (henon_config['t_base_10'], henon_config['t_base_2'])).astype(int)
henon_config['t_list'] = np.sort(henon_config['t_list'])

henon_config['t_diff'] = np.concatenate(
    ([henon_config['t_list'][0]], np.diff(henon_config['t_list']))).astype(int)

print(henon_config['t_diff'])

henon_config["x_sample"], henon_config["dx"] = np.linspace(
    henon_config["x_extents"][0],
    henon_config["x_extents"][1],
    henon_config["samples"],
    retstep=True
)

henon_config["y_sample"], henon_config["dy"] = np.linspace(
    henon_config["y_extents"][0],
    henon_config["y_extents"][1],
    henon_config["samples"],
    retstep=True
)

henon_config["xx"], henon_config["yy"] = np.meshgrid(
    henon_config["x_sample"],
    henon_config["y_sample"]
)

henon_config["x_flat"] = henon_config["xx"].flatten()
henon_config["y_flat"] = henon_config["yy"].flatten()
henon_config["px_flat"] = np.zeros_like(henon_config["x_flat"])
henon_config["py_flat"] = np.zeros_like(henon_config["x_flat"])

henon_config["total_samples"] = henon_config["x_flat"].size

henon_config["displacement"] = min(
    henon_config["dx"], henon_config["dy"]) * displacement_scale

displacement_table = np.array([sample_4d_displacement_on_a_sphere()
                              for _ in range(henon_config["total_samples"])])

henon_config["x_random_displacement"] = henon_config["x_flat"] + \
    displacement_table[:, 0] * henon_config["displacement"]
henon_config["y_random_displacement"] = henon_config["y_flat"] + \
    displacement_table[:, 1] * henon_config["displacement"]
henon_config["px_random_displacement"] = henon_config["px_flat"] + \
    displacement_table[:, 2] * henon_config["displacement"]
henon_config["py_random_displacement"] = henon_config["py_flat"] + \
    displacement_table[:, 3] * henon_config["displacement"]

henon_config["x_displacement"] = henon_config["x_flat"] + \
    henon_config["displacement"]
henon_config["y_displacement"] = henon_config["y_flat"] + \
    henon_config["displacement"]
henon_config["px_displacement"] = henon_config["px_flat"] + \
    henon_config["displacement"]
henon_config["py_displacement"] = henon_config["py_flat"] + \
    henon_config["displacement"]

# save lhc_config in the same directory of this script
with open(os.path.join(os.path.dirname(__file__), 'henon_config.pkl'), 'wb') as f:
    pickle.dump(henon_config, f)
