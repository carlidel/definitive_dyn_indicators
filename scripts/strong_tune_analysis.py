import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import os
import pathlib
import pickle

import henon_map_cpp as hm
from .config import henon_square_configuration as cfg

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

max_power_of_two = cfg["frequency_tracking"]["max_power_of_two"]
min_power_of_two = cfg["frequency_tracking"]["min_power_of_two"]

###############################################################################

x_sample = np.linspace(x_extents[0], x_extents[1], samples)
y_sample = np.linspace(y_extents[0], y_extents[1], samples)

xx, yy = np.meshgrid(x_sample, y_sample)

x_flat = xx.flatten()
y_flat = yy.flatten()
px_flat = np.zeros_like(x_flat)
py_flat = np.zeros_like(x_flat)

times_from = []
times_to = []
for i in range(min_power_of_two, max_power_of_two + 1):
    times_from.append(0)
    times_from.append(2**(i))
    times_to.append(2**(i))
    times_to.append(2**(i + 1))
times_from = np.asarray(times_from)
times_to = np.asarray(times_to)

max_turns = max(times_to)

print("Computing from {} to {} turns".format(
    2**min_power_of_two, max_turns))

settings_dictionary = {
    "min_power_of_two": min_power_of_two,
    "max_power_of_two": max_power_of_two,
    "samples": samples,
    "x_extents": x_extents,
    "y_extents": y_extents,
    "omega_x": omega_x,
    "omega_y": omega_y,
    "x_flat": x_flat,
    "y_flat": y_flat,
}

for epsilon in tqdm(epsilon_list):
    engine = hm.henon_tracker(x_flat, px_flat, y_flat, py_flat, omega_x, omega_y)

    birkhoff_tunes = engine.birkhoff_tunes(
        max_turns, epsilon, 0.0, from_idx=times_from, to_idx=times_to
    )
    fft_tunes = engine.fft_tunes(
        max_turns, epsilon, 0.0, from_idx=times_from, to_idx=times_to
    ) 

    with open(
        os.path.join(DATA_PATH, "henon_tunes_eps_{}.pkl".format(epsilon)), "wb"
    ) as f:
        pickle.dump(
            {
                "settings": settings_dictionary,
                "birkhoff_tunes": birkhoff_tunes,
                "fft_tunes": fft_tunes,
            },
            f,
        )
    
