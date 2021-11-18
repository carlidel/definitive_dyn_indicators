import numpy as np
import pandas as pd
import argparse
import os
import pathlib
import sys
import time
import pickle


from definitive_dyn_indicators.utils import xtrack_engine, get_lhc_mask
from definitive_dyn_indicators.scripts.config import lhc_square_test_configuration as lhc_config

OUTDIR = "/afs/cern.ch/work/c/camontan/public/definitive_dyn_indicators/data/"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LHC initial scan')
    parser.add_argument('-i', '--input', help='Input file', required=True)

    args = parser.parse_args()

    engine = xtrack_engine(context="CUDA", line_path=args.input)

    # Load data
    samples = lhc_config["samples"]

    x_flat = lhc_config["x_flat"]
    px_flat = lhc_config["px_flat"]
    y_flat = lhc_config["y_flat"]
    py_flat = lhc_config["py_flat"]
    t = lhc_config["long_tracking"]

    # Run engine
    x, px, y, py, steps = engine.track(x_flat, px_flat, y_flat, py_flat, t)

    # Format data
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
    output_filename = "initial_scan_{filename}.pkl"

    # Save data
    with open(os.path.join(OUTDIR, output_filename), 'wb') as f:
        pickle.dump({
            "line_name": filename, 
            "x": x, "px": px, "y": y, "py": py,
            "steps": steps
        }, f)