import numpy as np
import argparse
import os
import time
import pickle
import h5py
from tqdm import tqdm

from definitive_dyn_indicators.utils.xtrack_engine import xtrack_engine

OUTDIR = "/afs/cern.ch/work/c/camontan/public/definitive_dyn_indicators/data/"
ENGINEDIR = "/afs/cern.ch/work/c/camontan/public/definitive_dyn_indicators/data/"
CONFIG_DIR = "./"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LHC initial scan')
    parser.add_argument('-i', '--input', help='Input file', required=True)
    parser.add_argument('-c', '--continue-run',
                        help='Continue from previous run', action='store_true')
    parser.set_defaults(continue_run=False)
    parser.add_argument('-z', '--zeta-option',
                        help='Choose zeta option [0, 1, 2]', type=int, default=0,
                        choices=[0, 1, 2])
    parser.add_argument('-d', '--displacement-kind', 
                        help='Choose displacement kind', type=str, default="none",
                        choices=["none", "x", "y", "px", "py", "random"])
    parser.add_argument('-t', '--time-step',
                        help="kind of time step", type=str, default="basic",
                        choices=["basic", "advanced", "long", "advanced_full"])

    parser.add_argument('-e', '--engine', help='Engine directory', type=str,
                        default=ENGINEDIR),
    parser.add_argument('-o', '--output', help='Output directory', type=str,
                        default=OUTDIR)

    args = parser.parse_args()

    ENGINEDIR = args.engine
    OUTDIR = args.output

    # Load configuration
    with open(os.path.join(CONFIG_DIR, 'global_config.pkl'), 'rb') as f:
        lhc_config = pickle.load(f)

    # Load data
    print("Loading data.")
    x_flat = lhc_config["x_flat"]
    px_flat = lhc_config["px_flat"]
    y_flat = lhc_config["y_flat"]
    py_flat = lhc_config["py_flat"]
    
    if args.displacement_kind == "x":
        x_flat = lhc_config["x_x_displacement"]
    elif args.displacement_kind == "y":
        y_flat = lhc_config["y_y_displacement"]
    elif args.displacement_kind == "px":
        px_flat = lhc_config["px_px_displacement"]
    elif args.displacement_kind == "py":
        py_flat = lhc_config["py_py_displacement"]
    elif args.displacement_kind == "random":
        x_flat = lhc_config["x_random_displacement"]
        px_flat = lhc_config["px_random_displacement"]
        y_flat = lhc_config["y_random_displacement"]
        py_flat = lhc_config["py_random_displacement"]

    z_flat = np.ones_like(x_flat) * lhc_config[f"zeta_{args.zeta_option}"]

    print("x_flat size:", x_flat.shape)
    print("z_option:", lhc_config[f"zeta_{args.zeta_option}"])

    # Get filename from input
    filename = os.path.basename(args.input)
    # remove extension from filename
    filename = os.path.splitext(filename)[0]

    output_filename_engine = f"scan_{filename}_z_{args.zeta_option}_d_{args.displacement_kind}_tstep_{args.time_step}_engine.pkl"

    if not args.continue_run:
        print("Creating the engine.")
        engine = xtrack_engine(context="CUDA", line_path=args.input)
        current_t = 0
        iteration = 0
    else:
        print("Loading the engine")
        with open(os.path.join(ENGINEDIR, output_filename_engine), 'rb') as f:
            d = pickle.load(f)
            filename, current_t, iteration, engine = d["line_name"], d["current_t"], d["iteration"], d["engine"]

    samples = lhc_config["samples"]

    if args.time_step == "advanced_full":
        total_t = 0
        output_filename = f"scan_{filename}_z_{args.zeta_option}_d_{args.displacement_kind}_tstep_{args.time_step}_complete.hdf5"
        f = h5py.File(os.path.join(OUTDIR, output_filename), "w")

        dataset = f.create_dataset("config", data=np.array([42, 42]))
        dataset.attrs["line_name"] = filename
        dataset.attrs["zeta_option"] = args.zeta_option
        dataset.attrs["displacement_kind"] = args.displacement_kind
        dataset.attrs["time_step"] = args.time_step,

        for i, (t, total_t) in tqdm(enumerate(zip(lhc_config["t_diff"], lhc_config["t_list"])), total=len(lhc_config["t_diff"])):
            print(f"Executing {t} turns...")

            if i == 0:
                engine.track(x_flat, px_flat, y_flat, py_flat, t, zeta=z_flat)
            else:
                engine.keep_tracking(t)

            print("Saving data.")
            sorted_particles = engine.sort_particles()

            f[f"x/{total_t}"] = sorted_particles["x"].reshape(samples, -1)
            f[f"px/{total_t}"] = sorted_particles["px"].reshape(samples, -1)
            f[f"y/{total_t}"] = sorted_particles["y"].reshape(samples, -1)
            f[f"py/{total_t}"] = sorted_particles["py"].reshape(samples, -1)
            f[f"zeta/{total_t}"] = sorted_particles["zeta"].reshape(samples, -1)
            f[f"delta/{total_t}"] = sorted_particles["delta"].reshape(samples, -1)
            f[f"steps/{total_t}"] = sorted_particles["steps"].reshape(samples, -1)
            
            print("Total time:", total_t)

        f.close()
        print("Done.")
        current_t = total_t

    else:
        if args.time_step == "basic":
            t = lhc_config["tracking"]
        elif args.time_step == "long":
            t = lhc_config["long_tracking"]    
        else:
            if iteration < len(lhc_config["t_list"]):
                t = lhc_config["t_list"][iteration] - current_t
            else:
                print("No more time steps available.")
                print("Using basic tracking step.")
                t = lhc_config["tracking"]

        output_filename = f"scan_{filename}_z_{args.zeta_option}_d_{args.displacement_kind}_tstep_{args.time_step}_nturns_{current_t + t}.pkl"

        print("t iterations:", t)

        # Run engine
        print("Running the engine.")
        start = time.perf_counter()
        if not args.continue_run:
            engine.track(x_flat, px_flat, y_flat, py_flat, t, zeta=z_flat)
        else:
            engine.keep_tracking(t)
        sorted_particles = engine.sort_particles()
        end = time.perf_counter()

        # format time in hh:mm:ss
        time_elapsed = end - start
        hours, rem = divmod(time_elapsed, 3600)
        minutes, seconds = divmod(rem, 60)
        print("Time elapsed: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
        
        # Format data
        print("Formatting data.")
        x = sorted_particles["x"].reshape(samples, -1)
        px = sorted_particles["px"].reshape(samples, -1)
        y = sorted_particles["y"].reshape(samples, -1)
        py = sorted_particles["py"].reshape(samples, -1)
        zeta = sorted_particles["zeta"].reshape(samples, -1)
        delta = sorted_particles["delta"].reshape(samples, -1)
        steps = sorted_particles["steps"].reshape(samples, -1)

        # compose output filename
        print(f"Saving data to {output_filename}")
        # Save data
        with open(os.path.join(OUTDIR, output_filename), 'wb') as f:
            pickle.dump({
                "line_name": filename,
                "config": lhc_config,
                "zeta_option": args.zeta_option,
                "displacement_kind": args.displacement_kind,
                "time_step": args.time_step,
                "x": x, "px": px, "y": y, "py": py,
                "zeta": zeta, "delta": delta,
                "steps": steps
            }, f)

    # Save engine
    print(f"Saving engine to {output_filename_engine}")
    with open(os.path.join(ENGINEDIR, output_filename_engine), 'wb') as f:
        pickle.dump({
            "line_name": filename,
            "config": lhc_config,
            "zeta_option": args.zeta_option,
            "displacement_kind": args.displacement_kind,
            "time_step": args.time_step,
            "current_t": current_t + t,
            "iteration": iteration + 1,
            "engine": engine
        }, f)
