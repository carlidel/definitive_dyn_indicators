import numpy as np
import pathlib
import os
from tqdm.auto import tqdm
import pickle

import henon_map_cpp as hm
from config import henon_square_configuration as cfg


# get path of this script with pathlib
script_path = pathlib.Path(__file__).parent.absolute()
DATA_PATH = script_path.parent.joinpath('data')
print("DATA_PATH:", DATA_PATH)

###############################################################################

samples = cfg["samples"]
x_extents = cfg["x_extents"]
y_extents = cfg["y_extents"]
epsilon_list = cfg["epsilon_list"]

omega_x = cfg["omega_x"]
omega_y = cfg["omega_y"]

long_tracking = cfg["long_tracking"]

###############################################################################

x_sample = np.linspace(x_extents[0], x_extents[1], samples)
y_sample = np.linspace(y_extents[0], y_extents[1], samples)

xx, yy = np.meshgrid(x_sample, y_sample)

x_flat = xx.flatten()
y_flat = yy.flatten()
px_flat = np.zeros_like(x_flat)
py_flat = np.zeros_like(x_flat)

for epsilon in tqdm(epsilon_list):
    engine = hm.henon_tracker(x_flat, px_flat, y_flat,
                              py_flat, omega_x, omega_y)
    engine.track(long_tracking, epsilon, 0.0)

    steps = engine.get_steps()

    with open(DATA_PATH.joinpath(f"henon_long_tracking_{epsilon:.1f}.pkl"), 'wb') as f:
        pickle.dump(
            {
                "config": cfg,
                "x_flat": x_flat,
                "y_flat": y_flat,
                "steps": steps,
            },
            f
        )