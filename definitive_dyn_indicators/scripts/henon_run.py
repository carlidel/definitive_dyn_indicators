import numpy as np
import argparse
import os
import time
import datetime
import pickle
from tqdm import tqdm
import h5py
import pandas as pd
import shutil
import sys

from henon_map_cpp import henon_tracker


class fixed_henon(object):
    def __init__(self, omega_x, omega_y, epsilon=0.0, mu=0.0, barrier=10.0, kick_module=np.nan, modulation_kind="sps", omega_0=np.nan, max_t=100000000, force_CPU=False):
        self.omega_x = omega_x
        self.omega_y = omega_y
        self.epsilon = epsilon
        self.mu = mu
        self.barrier = barrier
        self.kick_module = kick_module
        self.modulation_kind = modulation_kind
        self.omega_0 = omega_0
        self.max_t = max_t
        self.force_CPU = force_CPU

        self.engine = None

    def create(self, x, px, y, py):
        self.engine = henon_tracker(x, px, y, py, self.force_CPU)
        self.engine.compute_a_modulation(
            self.max_t, self.omega_x, self.omega_y, self.epsilon,
            self.modulation_kind, self.omega_0, offset=0
        )

    def track(self, x, px, y, py, t):
        self.engine = henon_tracker(x, px, y, py, self.force_CPU)
        self.engine.compute_a_modulation(
            self.max_t, self.omega_x, self.omega_y, self.epsilon,
            self.modulation_kind, self.omega_0, offset=0
        )
        self.engine.track(t, self.mu, self.barrier, self.kick_module, False)

        return self.engine.get_x(), self.engine.get_px(), self.engine.get_y(), self.engine.get_py(), self.engine.get_steps()

    def track_megno(self, x, px, y, py, t_list):
        engine = henon_tracker(x, px, y, py, self.force_CPU)
        self.engine.compute_a_modulation(
            self.max_t, self.omega_x, self.omega_y, self.epsilon,
            self.modulation_kind, self.omega_0, offset=0
        )
        megno = engine.track_MEGNO(
            t_list, self.mu, self.barrier, self.kick_module)
        return megno

    def keep_tracking(self, t):
        assert(self.engine is not None)
        self.engine.track(t, self.mu, self.barrier, self.kick_module, False)

        return self.engine.get_x(), self.engine.get_px(), self.engine.get_y(), self.engine.get_py(), self.engine.get_steps()

    def track_and_reverse(self, x, px, y, py, t):
        engine = henon_tracker(x, px, y, py, self.force_CPU)
        self.engine.compute_a_modulation(
            self.max_t, self.omega_x, self.omega_y, self.epsilon,
            self.modulation_kind, self.omega_0, offset=0
        )
        engine.track(t, self.mu, self.barrier, self.kick_module, False)
        engine.track(t, self.mu, self.barrier, self.kick_module, True)

        return engine.get_x(), engine.get_px(), engine.get_y(), engine.get_py(), engine.get_steps()


def henon_run(omega_x, omega_y, modulation_kind, epsilon, mu, kick_module, omega_0, displacement_kind, tracking, outdir, henon_config, force_CPU=False):

    # Load data
    print("Loading data...")
    if tracking == "megno":
        x_flat = np.concatenate(
            (henon_config["x_flat"], henon_config["x_random_displacement"]))
        px_flat = np.concatenate(
            (henon_config["px_flat"], henon_config["px_random_displacement"]))
        y_flat = np.concatenate(
            (henon_config["y_flat"], henon_config["y_random_displacement"]))
        py_flat = np.concatenate(
            (henon_config["py_flat"], henon_config["py_random_displacement"]))
    else:
        x_flat = henon_config["x_flat"]
        px_flat = henon_config["px_flat"]
        y_flat = henon_config["y_flat"]
        py_flat = henon_config["py_flat"]

        if displacement_kind == "x":
            x_flat = henon_config["x_displacement"]
        elif displacement_kind == "y":
            y_flat = henon_config["y_displacement"]
        elif displacement_kind == "px":
            px_flat = henon_config["px_displacement"]
        elif displacement_kind == "py":
            py_flat = henon_config["py_displacement"]
        elif displacement_kind == "random":
            x_flat = henon_config["x_random_displacement"]
            y_flat = henon_config["y_random_displacement"]
            px_flat = henon_config["px_random_displacement"]
            py_flat = henon_config["py_random_displacement"]

    # Create engine
    print("Creating engine...")
    engine = fixed_henon(omega_x, omega_y, epsilon, mu,
                         henon_config["barrier"], kick_module,
                         modulation_kind, omega_0, force_CPU)

    # create hdf5 file
    data = h5py.File(outdir, "w")

    if tracking == "track":
        # start chronometer
        start = time.time()
        x, px, y, py, steps = engine.track(
            x_flat, px_flat, y_flat, py_flat, henon_config["extreme_tracking"])

        data["x"] = x
        data["px"] = px
        data["y"] = y
        data["py"] = py
        data["steps"] = steps
        # stop chronometer
        end = time.time()
        # print time in hh:mm:ss
        print(f"Elapsed time: {datetime.timedelta(seconds=end-start)}")
    elif tracking == "step_track":
        for i, (t, t_sum) in tqdm(enumerate(zip(henon_config["t_diff"], henon_config["t_list"])), total=len(henon_config["t_diff"])):
            if i == 0:
                x, px, y, py, steps = engine.track(
                    x_flat, px_flat, y_flat, py_flat, t)
            else:
                x, px, y, py, steps = engine.keep_tracking(t)

            data[f"x/{t_sum}"] = x
            data[f"px/{t_sum}"] = px
            data[f"y/{t_sum}"] = y
            data[f"py/{t_sum}"] = py

    elif tracking == "track_and_reverse":
        for i, (t, t_sum) in tqdm(enumerate(zip(henon_config["t_diff"], henon_config["t_list"])), total=len(henon_config["t_diff"])):
            x, px, y, py, steps = engine.track_and_reverse(
                x_flat, px_flat, y_flat, py_flat, t)

            data[f"x/{t_sum}"] = x
            data[f"px/{t_sum}"] = px
            data[f"y/{t_sum}"] = y
            data[f"py/{t_sum}"] = py

    elif tracking == "megno":
        # start chronometer
        start = time.time()
        megno = engine.track_megno(
            x_flat, px_flat, y_flat, py_flat, henon_config["t_list"])
        # stop chronometer
        end = time.time()
        # print time in hh:mm:ss
        print(f"Elapsed time: {datetime.timedelta(seconds=end-start)}")

        for i in range(len(henon_config["t_list"])):
            data[f"megno/{henon_config['t_list'][i]}"] = megno[i]

    elif tracking == "birkhoff_tunes":
        # start chronometer
        start = time.time()

        engine.create(x_flat, px_flat, y_flat, py_flat)
        from_idx = np.array(
            [0 for _ in henon_config["t_base_2"][:-1]] +
            [i for i in henon_config["t_base_2"][:-1]], dtype=int
        )
        to_idx = np.array(
            [i for i in henon_config["t_base_2"][:-1]] +
            [i * 2 for i in henon_config["t_base_2"][:-1]], dtype=int
        )
        tunes = engine.engine.birkhoff_tunes(
            henon_config["t_base_2"][-1],
            engine.epsilon, engine.mu, engine.barrier,
            engine.kick_module, engine.kick_sigma,
            engine.modulation_kind, engine.omega_0,
            from_idx=from_idx, to_idx=to_idx
        )
        for i in range(len(tunes)):
            data[f"tune_x/{tunes.iloc[i]['from']}/{tunes.iloc[i]['to']}"] = tunes.iloc[i]['tune_x']
            data[f"tune_y/{tunes.iloc[i]['from']}/{tunes.iloc[i]['to']}"] = tunes.iloc[i]['tune_y']

        # stop chronometer
        end = time.time()
        # print time in hh:mm:ss
        print(f"Time elapsed: {datetime.timedelta(seconds=end-start)}")

    elif tracking == "fft_tunes":
        # start chronometer
        start = time.time()
        engine.create(x_flat, px_flat, y_flat, py_flat)
        from_idx = np.array(
            [0 for _ in henon_config["t_base_2"][:-1]] +
            [i for i in henon_config["t_base_2"][:-1]], dtype=int
        )
        to_idx = np.array(
            [i for i in henon_config["t_base_2"][:-1]] +
            [i * 2 for i in henon_config["t_base_2"][:-1]], dtype=int
        )
        tunes = engine.engine.fft_tunes(
            henon_config["t_base_2"][-1],
            engine.epsilon, engine.mu, engine.barrier,
            engine.kick_module, engine.kick_sigma,
            engine.modulation_kind, engine.omega_0,
            from_idx=from_idx, to_idx=to_idx
        )
        for i in range(len(tunes)):
            data[f"tune_x/{tunes.iloc[i]['from']}/{tunes.iloc[i]['to']}"] = tunes.iloc[i]['tune_x']
            data[f"tune_y/{tunes.iloc[i]['from']}/{tunes.iloc[i]['to']}"] = tunes.iloc[i]['tune_y']
        # stop chronometer
        end = time.time()
        # print time in hh:mm:ss
        print(f"Time elapsed: {datetime.timedelta(seconds=end-start)}")

    # create new dataset in file
    dataset = data.create_dataset("config", data=np.array([42, 42]))

    # fill attributes of dataset
    dataset.attrs["omega_x"] = omega_x
    dataset.attrs["omega_y"] = omega_y
    dataset.attrs["epsilon"] = epsilon
    dataset.attrs["mu"] = mu
    dataset.attrs["barrier"] = henon_config["barrier"]
    dataset.attrs["kick_module"] = kick_module
    dataset.attrs["modulation_kind"] = modulation_kind
    dataset.attrs["omega_0"] = omega_0
    dataset.attrs["tracking"] = tracking
    dataset.attrs["displacement_kind"] = displacement_kind

    data.close()