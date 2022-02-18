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
        print("Tracking at t = {}".format(t))
        self.engine = henon_tracker(x, px, y, py, self.force_CPU)
        print("Modulation")
        self.engine.compute_a_modulation(
            self.max_t, self.omega_x, self.omega_y, self.epsilon,
            self.modulation_kind, self.omega_0, offset=0
        )
        self.engine.track(t, self.mu, self.barrier, self.kick_module, False)
        print("Tracking called, returning...")
        return self.engine.get_x(), self.engine.get_px(), self.engine.get_y(), self.engine.get_py(), self.engine.get_steps()

    def track_megno(self, x, px, y, py, t_list):
        engine = henon_tracker(x, px, y, py, self.force_CPU)
        engine.compute_a_modulation(
            self.max_t, self.omega_x, self.omega_y, self.epsilon,
            self.modulation_kind, self.omega_0, offset=0
        )
        megno = engine.track_MEGNO(
            t_list, self.mu, self.barrier, self.kick_module)
        return megno

    def track_realignments(self, x, px, y, py, t_list, m_low, m_barrier):
        engine = henon_tracker(x, px, y, py, self.force_CPU)
        engine.compute_a_modulation(
            self.max_t, self.omega_x, self.omega_y, self.epsilon,
            self.modulation_kind, self.omega_0, offset=0
        )
        megno = engine.track_realignments(
            t_list, self.mu, self.barrier, self.kick_module, m_low, m_barrier)
        return megno

    def track_tangent_map(self, x, px, y, py, t_list):
        engine = henon_tracker(x, px, y, py, self.force_CPU)
        engine.compute_a_modulation(
            self.max_t, self.omega_x, self.omega_y, self.epsilon,
            self.modulation_kind, self.omega_0, offset=0
        )
        tm = engine.track_tangent_map(
            t_list, self.mu, self.barrier, self.kick_module)
        return tm

    def keep_tracking(self, t):
        assert(self.engine is not None)
        self.engine.track(t, self.mu, self.barrier, self.kick_module, False)

        return self.engine.get_x(), self.engine.get_px(), self.engine.get_y(), self.engine.get_py(), self.engine.get_steps()

    def track_and_reverse(self, t):
        assert(self.engine is not None)
        self.engine.reset()
        self.engine.track(t, self.mu, self.barrier, self.kick_module, False)
        self.engine.track(t, self.mu, self.barrier, self.kick_module, True)

        return self.engine.get_x(), self.engine.get_px(), self.engine.get_y(), self.engine.get_py(), self.engine.get_steps()


def henon_run(omega_x, omega_y, modulation_kind, epsilon, mu, kick_module, omega_0, displacement_kind, tracking, outdir, henon_config, force_CPU=False):

    # Load data
    print("Loading data...")
    if tracking == "megno" or tracking == "true_displacement":
        if displacement_kind == "random":
            x_flat = np.concatenate(
                (henon_config["x_flat"], henon_config["x_random_displacement"]))
            px_flat = np.concatenate(
                (henon_config["px_flat"], henon_config["px_random_displacement"]))
            y_flat = np.concatenate(
                (henon_config["y_flat"], henon_config["y_random_displacement"]))
            py_flat = np.concatenate(
                (henon_config["py_flat"], henon_config["py_random_displacement"]))
        elif displacement_kind == "x":
            x_flat = np.concatenate(
                (henon_config["x_flat"], henon_config["x_displacement"]))
            px_flat = np.concatenate(
                (henon_config["px_flat"], henon_config["px_flat"]))
            y_flat = np.concatenate(
                (henon_config["y_flat"], henon_config["y_flat"]))
            py_flat = np.concatenate(
                (henon_config["py_flat"], henon_config["py_flat"]))
        elif displacement_kind == "y":
            x_flat = np.concatenate(
                (henon_config["x_flat"], henon_config["x_flat"]))
            px_flat = np.concatenate(
                (henon_config["px_flat"], henon_config["px_flat"]))
            y_flat = np.concatenate(
                (henon_config["y_flat"], henon_config["y_displacement"]))
            py_flat = np.concatenate(
                (henon_config["py_flat"], henon_config["py_flat"]))
        elif displacement_kind == "px":
            x_flat = np.concatenate(
                (henon_config["x_flat"], henon_config["x_flat"]))
            px_flat = np.concatenate(
                (henon_config["px_flat"], henon_config["px_displacement"]))
            y_flat = np.concatenate(
                (henon_config["y_flat"], henon_config["y_flat"]))
            py_flat = np.concatenate(
                (henon_config["py_flat"], henon_config["py_flat"]))
        elif displacement_kind == "py":
            x_flat = np.concatenate(
                (henon_config["x_flat"], henon_config["x_flat"]))
            px_flat = np.concatenate(
                (henon_config["px_flat"], henon_config["px_flat"]))
            y_flat = np.concatenate(
                (henon_config["y_flat"], henon_config["y_flat"]))
            py_flat = np.concatenate(
                (henon_config["py_flat"], henon_config["py_displacement"]))
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
    print("Instatiating engine...")
    engine = fixed_henon(omega_x, omega_y, epsilon, mu,
                         henon_config["barrier"], kick_module,
                         modulation_kind, omega_0,
                         max_t=henon_config["extreme_tracking"], force_CPU=force_CPU)

    # create hdf5 file

    filename = f"henon_ox_{omega_x}_oy_{omega_y}_modulation_{modulation_kind}_eps_{epsilon}_mu_{mu}_kmod_{kick_module}_o0_{omega_0}_disp_{displacement_kind}_data_{tracking}.hdf5"

    data = h5py.File(os.path.join(outdir, filename) , "w")

    if tracking == "track":
        # start chronometer
        start = time.time()
        print("Tracking...")
        x, px, y, py, steps = engine.track(
            x_flat, px_flat, y_flat, py_flat, henon_config["extreme_tracking"])

        data.create_dataset("x", data=x, compression="gzip", shuffle=True)
        data.create_dataset("px", data=px, compression="gzip", shuffle=True)
        data.create_dataset("y", data=y, compression="gzip", shuffle=True)
        data.create_dataset("py", data=py, compression="gzip", shuffle=True)
        data.create_dataset("steps", data=steps, compression="gzip", shuffle=True)

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

            data.create_dataset(f"x/{t_sum}", data=x, compression="gzip", shuffle=True)
            data.create_dataset(f"px/{t_sum}", data=px, compression="gzip", shuffle=True)
            data.create_dataset(f"y/{t_sum}", data=y, compression="gzip", shuffle=True)
            data.create_dataset(f"py/{t_sum}", data=py, compression="gzip", shuffle=True)
        
        data.create_dataset("steps", data=steps, compression="gzip", shuffle=True)
    elif tracking == "track_and_reverse":
        print("Creating engine")
        engine.create(x_flat, px_flat, y_flat, py_flat)
        for i, (t, t_sum) in tqdm(enumerate(zip(henon_config["t_list"], henon_config["t_list"])), total=len(henon_config["t_list"])):
            x, px, y, py, steps = engine.track_and_reverse(t)

            data.create_dataset(f"x/{t_sum}", data=x, compression="gzip", shuffle=True)
            data.create_dataset(f"px/{t_sum}", data=px, compression="gzip", shuffle=True)
            data.create_dataset(f"y/{t_sum}", data=y, compression="gzip", shuffle=True)
            data.create_dataset(f"py/{t_sum}", data=py, compression="gzip", shuffle=True)

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
            data.create_dataset(f"megno/{henon_config['t_list'][i]}", data=megno[i], compression="gzip", shuffle=True)
    
    elif tracking == "true_displacement":
        # start chronometer
        start = time.time()
        displacement = engine.track_realignments(
            x_flat, px_flat, y_flat, py_flat, henon_config["t_list"], henon_config["displacement"], henon_config["displacement"]*1e3)
        # stop chronometer
        end = time.time()
        # print time in hh:mm:ss
        print(f"Elapsed time: {datetime.timedelta(seconds=end-start)}")

        for i in range(len(henon_config["t_list"])):
            data.create_dataset(f"displacement/{henon_config['t_list'][i]}", data=displacement[i], compression="gzip", shuffle=True)

        data.create_dataset("steps", data=engine.engine.get_steps(),
                            compression="gzip", shuffle=True)

    elif tracking == "tangent_map":
        # start chronometer
        start = time.time()
        tm = engine.track_tangent_map(
            x_flat, px_flat, y_flat, py_flat, henon_config["t_list"])
        # stop chronometer
        end = time.time()
        # print time in hh:mm:ss
        print(f"Elapsed time: {datetime.timedelta(seconds=end-start)}")

        for i in range(len(henon_config["t_list"])):
            data.create_dataset(f"tangent_map/{henon_config['t_list'][i]}", data=tm[i], compression="gzip", shuffle=True)

    elif tracking == "birkhoff_tunes":
        # start chronometer
        start = time.time()
        engine.force_CPU = True
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
            engine.mu, engine.barrier,
            engine.kick_module, from_idx=from_idx, to_idx=to_idx
        )
        for i in range(len(tunes)):
            data.create_dataset(f"tune_x/{tunes.iloc[i]['from']}/{tunes.iloc[i]['to']}", data=tunes.iloc[i]['tune_x'], compression="gzip", shuffle=True)
            data.create_dataset(f"tune_y/{tunes.iloc[i]['from']}/{tunes.iloc[i]['to']}", data=tunes.iloc[i]['tune_y'], compression="gzip", shuffle=True)

        # stop chronometer
        end = time.time()
        # print time in hh:mm:ss
        print(f"Time elapsed: {datetime.timedelta(seconds=end-start)}")

    elif tracking == "fft_tunes":
        # start chronometer
        start = time.time()
        engine.force_CPU = True
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
            engine.mu, engine.barrier,
            engine.kick_module, from_idx=from_idx, to_idx=to_idx
        )
        for i in range(len(tunes)):
            data.create_dataset(f"tune_x/{tunes.iloc[i]['from']}/{tunes.iloc[i]['to']}", data=tunes.iloc[i]['tune_x'], compression="gzip", shuffle=True)
            data.create_dataset(f"tune_y/{tunes.iloc[i]['from']}/{tunes.iloc[i]['to']}", data=tunes.iloc[i]['tune_y'], compression="gzip", shuffle=True)
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
