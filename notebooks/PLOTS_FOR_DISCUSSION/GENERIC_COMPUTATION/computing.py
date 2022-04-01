import argparse

# init argparse
parser = argparse.ArgumentParser(description='Henon Run')

# add arguments
parser.add_argument('--gpu_id', type=str, default="2", help='GPU to use')
parser.add_argument('--samples', type=int, default=100, help='number of points')
parser.add_argument('--data-dir', type=str, default='data/', help='data directory')
parser.add_argument('--t-ground-truth', type=int, default=100000000, help='ground truth t')
parser.add_argument('--t-normalization', type=int, default=1000, help='norm t')
parser.add_argument('--t-dynamic', type=int, default=1000000, help='arrival for dynamic indicators t')
parser.add_argument('--omega_x', type=float, default=0.310, help='omega_x')
parser.add_argument('--omega_y', type=float, default=0.320, help='omega_y')
parser.add_argument('--modulation-kind', type=str, default='sps', help='modulation kind')
parser.add_argument('--epsilon', type=float, default=0.0, help='epsilon')
parser.add_argument('--mu', type=float, default=0.0, help='mu')
parser.add_argument('--max_x', type=float, default=0.23, help='max_x')
parser.add_argument('--max_y', type=float, default=0.23, help='max_y')

# parse arguments
args = parser.parse_args()

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu_id)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import definitive_dyn_indicators.scripts.data_manager as dm
import os
from tqdm.auto import tqdm
from numba import njit
import lmfit
import joblib
from joblib import Parallel, delayed
import pandas as pd

njobs = os.cpu_count()
print(f'Number of cores: {njobs}')

# %%

DATA_DIR = args.data_dir
SAMPLES = args.samples

T_GROUND_TRUTH = args.t_ground_truth
T_NORMALIZATION = args.t_normalization
T_DYN_INDICATORS = args.t_dynamic

group = (
    args.omega_x,           # omega_x
    args.omega_y,           # omega_y
    args.modulation_kind,   # modulation_kind
    args.epsilon,           # epsilon
    args.mu,                # mu
    np.nan,                 # kick amplitude
    np.nan,                 # omega_0
)

group_random = tuple([group[i] if i!=5 else 1e-10 for i in range(len(group))])

X_EXTENTS = [0.0, args.max_x]
Y_EXTENTS = [0.0, args.max_y]

# %%

data = dm.data_manager(data_dir=DATA_DIR)

data.henon_config["samples"] = SAMPLES
data.henon_config["x_extents"] = X_EXTENTS
data.henon_config["y_extents"] = Y_EXTENTS

data.henon_config["t_base_2"] = np.array([], dtype=int)
data.henon_config["t_base"] = np.array([], dtype=int)
data.henon_config["t_base_10"] = np.logspace(
    int(np.log10(T_NORMALIZATION)),
    int(np.log10(T_GROUND_TRUTH)),
    int(np.log10(T_GROUND_TRUTH)) - int(np.log10(T_NORMALIZATION)) + 1,
    base=10, dtype=int)
data.henon_config["t_linear"] = np.array([], dtype=int)

data.henon_config = dm.refresh_henon_config(data.henon_config)

config = data.get_config()
extents = config["x_extents"] + config["y_extents"]
samples = config["samples"]

times = np.asarray(data.get_times())

# %%

full_lyapunov = data.fast_lyapunov_indicator(group)

# %%

data_tune = dm.data_manager(
    data_dir=os.path.join(DATA_DIR, "10_6"))

data_tune.henon_config["samples"] = SAMPLES
data_tune.henon_config["x_extents"] = X_EXTENTS
data_tune.henon_config["y_extents"] = Y_EXTENTS

# find the closest power of 2 to T_DYN_INDICATORS
t_tune_pow = np.ceil(np.log2(T_DYN_INDICATORS)) + 2
t_tune_min = 4

data_tune.henon_config["t_base_2"] = 2 ** np.arange(
    t_tune_min, t_tune_pow + 1, dtype=int)
data_tune.henon_config["t_base"] = np.array([], dtype=int)
data_tune.henon_config["t_base_10"] = np.array([], dtype=int)
data_tune.henon_config["t_linear"] = np.array([], dtype=int)

data_tune.henon_config = dm.refresh_henon_config(data_tune.henon_config)

times_tune = np.asarray(data_tune.get_times())
 # %%

tunes = data_tune.birkhoff_tunes(group)

# %%

data_minor = dm.data_manager(
    data_dir=os.path.join(DATA_DIR, "10_6"))

data_minor.henon_config["samples"] = SAMPLES
data_minor.henon_config["x_extents"] = X_EXTENTS
data_minor.henon_config["y_extents"] = Y_EXTENTS

data_minor.henon_config["t_base_2"] = np.array([], dtype=int)
data_minor.henon_config["t_base"] = np.arange(
    T_NORMALIZATION, T_DYN_INDICATORS + T_NORMALIZATION, T_NORMALIZATION, dtype=int)

data_minor.henon_config["t_base_10"] = np.array([], dtype=int)
data_minor.henon_config["t_linear"] = np.array([], dtype=int)

data_minor.henon_config = dm.refresh_henon_config(data_minor.henon_config)

times_minor = np.asarray(data_minor.get_times())

# %%

gali = data_minor.global_alignment_index(group)
sali = data_minor.smallest_alignment_index(group)
lyapunov = data_minor.fast_lyapunov_indicator(group)
ortho_lyap_max, ortho_lyap_avg = data_minor.orthogonal_lyapunov_indicator(group)

reverse = data_minor.reversibility_error(group)
