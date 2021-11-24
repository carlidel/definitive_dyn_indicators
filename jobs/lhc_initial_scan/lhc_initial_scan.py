import numpy as np
import pandas as pd
import argparse
import os
import pathlib
import sys
import time
import pickle


from definitive_dyn_indicators.utils.xtrack_engine import xtrack_engine
from definitive_dyn_indicators.scripts.config import lhc_square_test_configuration as lhc_config

OUTDIR = "/afs/cern.ch/work/c/camontan/public/definitive_dyn_indicators/data/"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LHC initial scan')
    parser.add_argument('-i', '--input', help='Input file', required=True)

    parser.add_argument('--short-track', help='Short track', action='store_true', dest='short_track')
    parser.add_argument('--long-track', help='Long track', action='store_false', dest='short_track')
    parser.set_defaults(short_track=False)

    args = parser.parse_args()

    print("Creating the engine.")
    engine = xtrack_engine(context="CUDA", line_path=args.input)

    # Load data
    print("Loading data.")
    samples = lhc_config["samples"]

    x_flat = lhc_config["x_flat"]
    px_flat = lhc_config["px_flat"]
    y_flat = lhc_config["y_flat"]
    py_flat = lhc_config["py_flat"]
    t = lhc_config["long_tracking"] if not args.short_track else lhc_config["short_tracking"]

    print("x_flat size:", x_flat.shape)
    print("px_flat size:", px_flat.shape)
    print("y_flat size:", y_flat.shape)
    print("py_flat size:", py_flat.shape)

    # Run engine
    print("Running the engine.")
    start = time.perf_counter()
    x, px, y, py, steps = engine.track(x_flat, px_flat, y_flat, py_flat, t)
    end = time.perf_counter()
    print(f"Time: {end - start}")
    # Format data
    print("Formatting data.")
    x = x.reshape(samples, -1)
    px = px.reshape(samples, -1)
    y = y.reshape(samples, -1)
    py = py.reshape(samples, -1)
    steps = steps.reshape(samples, -1)

    # Get filename from input
    filename = os.path.basename(args.input)
    # remove extension from filename
    filename = os.path.splitext(filename)[0]
    # compose output filename
    lenght_label = "short" if args.short_track else "long"
    output_filename = f"initial_scan_{filename}_{lenght_label}.pkl"
    print(f"Saving data to {output_filename}")

    # Save data
    with open(os.path.join(OUTDIR, output_filename), 'wb') as f:
        pickle.dump({
            "line_name": filename, 
            "x": x, "px": px, "y": y, "py": py,
            "steps": steps
        }, f)