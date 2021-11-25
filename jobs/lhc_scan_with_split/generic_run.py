import numpy as np
import pandas as pd
import argparse
import os
import pathlib
import sys
import time
import pickle


from definitive_dyn_indicators.utils.xtrack_engine import xtrack_engine
from .config import lhc_config

OUTDIR = "/afs/cern.ch/work/c/camontan/public/definitive_dyn_indicators/data/"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LHC initial scan')
    parser.add_argument('-i', '--input', help='Input file', required=True)

    parser.add_argument('-c', '--continue-run', help='Continue from previous run', action='store_true')
    parser.set_defaults(continue_run=False)

    args = parser.parse_args()

    if not args.continue_run:
        print("Creating the engine.")
        engine = xtrack_engine(context="CUDA", line_path=args.input)

        # Load data
        print("Loading data.")
        x_flat = lhc_config["x_flat"]
        px_flat = lhc_config["px_flat"]
        y_flat = lhc_config["y_flat"]
        py_flat = lhc_config["py_flat"]

        current_t = 0

        # Get filename from input
        filename = os.path.basename(args.input)
        # remove extension from filename
        filename = os.path.splitext(filename)[0]

    else:
        print("Loading the engine")
        with open(args.input, 'rb') as f:
            filename, current_t, engine = pickle.load(f)

    samples = lhc_config["samples"]
    t = lhc_config["tracking"]

    print("x_flat size:", x_flat.shape)
    print("px_flat size:", px_flat.shape)
    print("y_flat size:", y_flat.shape)
    print("py_flat size:", py_flat.shape)
    print("t value:", t)

    # Run engine
    print("Running the engine.")
    start = time.perf_counter()
    if not args.continue_run:
        x, px, y, py, steps = engine.track(x_flat, px_flat, y_flat, py_flat, t)
    else:
        x, px, y, py, steps = engine.keep_tracking(t)
    end = time.perf_counter()

    # format time in hh:mm:ss
    time_elapsed = end - start
    hours, rem = divmod(time_elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Time elapsed: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
    
    # Format data
    print("Formatting data.")
    x = x.reshape(samples, -1)
    px = px.reshape(samples, -1)
    y = y.reshape(samples, -1)
    py = py.reshape(samples, -1)
    steps = steps.reshape(samples, -1)

    # compose output filename
    output_filename = f"scan_{filename}_nturns_{current_t + t}.pkl"
    print(f"Saving data to {output_filename}")
    # Save data
    with open(os.path.join(OUTDIR, output_filename), 'wb') as f:
        pickle.dump({
            "line_name": filename,
            "x": x, "px": px, "y": y, "py": py,
            "steps": steps
        }, f)

    # Save engine
    output_filename_engine = f"scan_{filename}_engine.pkl"
    print(f"Saving engine to {output_filename_engine}")
    with open(os.path.join(OUTDIR, output_filename_engine), 'wb') as f:
        pickle.dump({
            "line_name": filename,
            "current_t": current_t + t,
            "engine": engine
        }, f)
