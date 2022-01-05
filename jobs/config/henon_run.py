import numpy as np
import argparse
import os
import time
import pickle
from tqdm import tqdm

from henon_map_cpp import henon_tracker

OUTDIR = "/afs/cern.ch/work/c/camontan/public/definitive_dyn_indicators/data/"
CONFIG_DIR = "./"

class fixed_henon(object):
    def __init__(self, omega_x, omega_y, epsilon=0.0, mu=0.0, barrier=10.0, kick_module=np.nan, kick_sigma=np.nan, modulation_kind="sps", omega_0=np.nan, force_CPU=False):
        self.omega_x = omega_x
        self.omega_y = omega_y
        self.epsilon = epsilon
        self.mu = mu
        self.barrier = barrier
        self.kick_module = kick_module
        self.kick_sigma = kick_sigma
        self.modulation_kind = modulation_kind
        self.omega_0 = omega_0
        self.force_CPU = force_CPU

        self.engine = None

    def create(self, x, px, y, py):
        self.engine = henon_tracker(x, px, y, py, self.omega_x,
                                    self.omega_y, self.force_CPU)

    def track(self, x, px, y, py, t):
        self.engine = henon_tracker(x, px, y, py, self.omega_x,
                                    self.omega_y, self.force_CPU)

        self.engine.track(t, self.epsilon, self.mu, self.barrier, self.kick_module,
                          self.kick_sigma, False, self.modulation_kind, self.omega_0)

        #print(engine.get_x())
        return self.engine.get_x(), self.engine.get_px(), self.engine.get_y(), self.engine.get_py(), self.engine.get_steps()

    def keep_tracking(self, t):
        assert(self.engine is not None)
        self.engine.track(t, self.epsilon, self.mu, self.barrier, self.kick_module,
                          self.kick_sigma, False, self.modulation_kind, self.omega_0)

        return self.engine.get_x(), self.engine.get_px(), self.engine.get_y(), self.engine.get_py(), self.engine.get_steps()

    def track_and_reverse(self, x, px, y, py, t):
        engine = henon_tracker(x, px, y, py, self.omega_x,
                               self.omega_y, self.force_CPU)

        engine.track(t, self.epsilon, self.mu, self.barrier, self.kick_module,
                     self.kick_sigma, False, self.modulation_kind, self.omega_0)
        engine.track(t, self.epsilon, self.mu, self.barrier, self.kick_module,
                     self.kick_sigma, True, self.modulation_kind, self.omega_0)

        return engine.get_x(), engine.get_px(), engine.get_y(), engine.get_py(), engine.get_steps()


if __name__ == '__main__':
    # Load configuration
    with open(os.path.join(CONFIG_DIR, 'henon_config.pkl'), 'rb') as f:
        henon_config = pickle.load(f)

    parser = argparse.ArgumentParser(description='Run Henon map')
    parser.add_argument('--omega-x', type=float, default=0.168)
    parser.add_argument('--omega-y', type=float, default=0.201)
    parser.add_argument('--epsilon', type=float, default=0.0)
    parser.add_argument('--mu', type=float, default=0.0)
    parser.add_argument('--barrier', type=float, default=10.0)
    parser.add_argument('--kick-module', type=float, default=np.nan)
    parser.add_argument('--kick-sigma', type=float, default=np.nan)
    parser.add_argument('--modulation-kind', type=str, default="sps")
    parser.add_argument('--omega-0', type=float, default=np.nan)
    parser.add_argument('--force-CPU', action='store_true')
    parser.add_argument('--outdir', type=str, default=OUTDIR)
    parser.add_argument('-d', '--displacement-kind',
                        help='Choose displacement kind', type=str, default="none",
                        choices=["none", "x", "y", "px", "py", "random"])
    parser.add_argument('--tracking', type=str, default="track",
                        choices=["track", "step_track", "track_and_reverse",
                                 "fft_tunes", "birkhoff_tunes"])


    args = parser.parse_args()

    OUTDIR = args.outdir
    filename = f"henon_ox_{args.omega_x}_oy_{args.omega_y}_modulation_{args.modulation_kind}_eps_{args.epsilon}_mu_{args.mu}_kmod_{args.kick_module}_ksig_{args.kick_sigma}_o0_{args.omega_0}_disp_{args.displacement_kind}_data_{args.tracking}.pkl"

    # Load data
    print("Loading data...")
    x_flat = henon_config["x_flat"]
    px_flat = henon_config["px_flat"]
    y_flat = henon_config["y_flat"]
    py_flat = henon_config["py_flat"]

    if args.displacement_kind == "x":
        x_flat = henon_config["x_displacement"]
    elif args.displacement_kind == "y":
        y_flat = henon_config["y_displacement"]
    elif args.displacement_kind == "px":
        px_flat = henon_config["px_displacement"]
    elif args.displacement_kind == "py":
        py_flat = henon_config["py_displacement"]
    elif args.displacement_kind == "random":
        x_flat = henon_config["x_random_displacement"]
        y_flat = henon_config["y_random_displacement"]
        px_flat = henon_config["px_random_displacement"]
        py_flat = henon_config["py_random_displacement"]

    # Create engine
    print("Creating engine...")
    engine = fixed_henon(args.omega_x, args.omega_y, args.epsilon, args.mu,
                         args.barrier, args.kick_module, args.kick_sigma,
                         args.modulation_kind, args.omega_0, args.force_CPU)

    data = {
        "omega_x": args.omega_x,
        "omega_y": args.omega_y,
        "epsilon": args.epsilon,
        "mu": args.mu,
        "barrier": args.barrier,
        "kick_module": args.kick_module,
        "kick_sigma": args.kick_sigma,
        "modulation_kind": args.modulation_kind,
        "omega_0": args.omega_0,

        "tracking": args.tracking,
    }

    if args.tracking == "track":
        x, px, y, py, steps = engine.track(x_flat, px_flat, y_flat, py_flat, henon_config["tracking"])

        data["x"] = x
        data["px"] = px
        data["y"] = y
        data["py"] = py
        data["steps"] = steps
    elif args.tracking == "step_track":
        data["x"] = {}
        data["px"] = {}
        data["y"] = {}
        data["py"] = {}
        for i, (t, t_sum) in tqdm(enumerate(zip(henon_config["t_diff"], henon_config["t_list"])), total=len(henon_config["t_diff"])):
            if i == 0:
                x, px, y, py, steps = engine.track(x_flat, px_flat, y_flat, py_flat, t)
            else:
                x, px, y, py, steps = engine.keep_tracking(t)
            
            data["x"][t_sum] = x
            data["px"][t_sum] = px
            data["y"][t_sum] = y
            data["py"][t_sum] = py
    elif args.tracking == "track_and_reverse":
        data["x"] = {}
        data["px"] = {}
        data["y"] = {}
        data["py"] = {}
        for i, (t, t_sum) in tqdm(enumerate(zip(henon_config["t_diff"], henon_config["t_list"])), total=len(henon_config["t_diff"])):
            x, px, y, py, steps = engine.track_and_reverse(
                x_flat, px_flat, y_flat, py_flat, t)

            data["x"][t_sum] = x
            data["px"][t_sum] = px
            data["y"][t_sum] = y
            data["py"][t_sum] = py
    elif args.tracking == "birkhoff_tunes":
        engine.create(x_flat, px_flat, y_flat, py_flat)
        from_idx = np.array(
            [0 for _ in henon_config["t_base_2"]] +
            [i for i in henon_config["t_base_2"][:-1]], dtype=int
        )
        to_idx = np.array(
            [i for i in henon_config["t_base_2"]] +
            [i * 2 for i in henon_config["t_base_2"][:-1]], dtype=int
        )
        tunes = engine.engine.birkhoff_tunes(
            henon_config["t_base_2"][-1],
            engine.epsilon, engine.mu, engine.barrier,
            engine.kick_module, engine.kick_sigma,
            engine.modulation_kind, engine.omega_0,
            from_idx=from_idx, to_idx=to_idx
        )
        data["tunes"] = tunes
    elif args.tracking == "fft_tunes":
        engine.create(x_flat, px_flat, y_flat, py_flat)
        from_idx = np.array(
            [0 for _ in henon_config["t_base_2"]] +
            [i for i in henon_config["t_base_2"][:-1]], dtype=int
        )
        to_idx = np.array(
            [i for i in henon_config["t_base_2"]] +
            [i * 2 for i in henon_config["t_base_2"][:-1]], dtype=int
        )
        tunes = engine.engine.fft_tunes(
            henon_config["t_base_2"][-1],
            engine.epsilon, engine.mu, engine.barrier,
            engine.kick_module, engine.kick_sigma,
            engine.modulation_kind, engine.omega_0,
            from_idx=from_idx, to_idx=to_idx
        )
        data["tunes"] = tunes

    # Save data
    print(f"Saving data to {os.path.join(OUTDIR + filename)}...")
    with open(os.path.join(OUTDIR + filename), "wb") as f:
        pickle.dump(data, f)