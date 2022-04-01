from fileinput import filename
import numpy as np
import pandas as pd
import h5py
import os
import pickle
from tqdm import tqdm
from numba import njit, prange
from zmq import THREAD_SCHED_POLICY
from . import dynamic_indicators as di
from . import henon_run as hr
import matplotlib.pyplot as plt
import matplotlib
import lmfit
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema
import joblib


from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

N_THREADS = os.cpu_count()
RANDOM_SEED = 42
rs = RandomState(MT19937(SeedSequence(RANDOM_SEED)))

def sample_4d_displacement_on_a_sphere():
    n = rs.normal(0, 1, size=4)
    module = np.sqrt(np.sum(n ** 2))
    return n / module

HENON_BASE_CONFIG = {
    'name': 'henon_square_universal_config',
    'samples': 100,
    'random_seed': RANDOM_SEED,
    'displacement_scale': 1e-10,

    'x_extents': [0.0, 0.8],
    'y_extents': [0.0, 0.8],

    'extreme_tracking': 100000000,

    't_base_10': np.logspace(1, 6, 51, dtype=int),
    't_base_2': np.logspace(1, 20, 20, dtype=int, base=2),
    't_linear': np.linspace(1000, 1000000, 1000, dtype=int),
    't_base': np.arange(1, 1001, 1, dtype=int),

    'barrier': 10.0
}

def refresh_henon_config(henon_config):
    henon_config['t_list'] = np.concatenate(
        (henon_config['t_base_10'], henon_config['t_base_2'], henon_config['t_linear'], henon_config['t_base'])).astype(int)
    henon_config['t_list'] = np.unique(henon_config['t_list']).astype(int)

    henon_config['t_diff'] = np.concatenate(
        ([henon_config['t_list'][0]], np.diff(henon_config['t_list']))).astype(int)

    henon_config["x_sample"], henon_config["dx"] = np.linspace(
        henon_config["x_extents"][0],
        henon_config["x_extents"][1],
        henon_config["samples"],
        retstep=True
    )

    henon_config["y_sample"], henon_config["dy"] = np.linspace(
        henon_config["y_extents"][0],
        henon_config["y_extents"][1],
        henon_config["samples"],
        retstep=True
    )

    henon_config["xx"], henon_config["yy"] = np.meshgrid(
        henon_config["x_sample"],
        henon_config["y_sample"]
    )

    henon_config["x_flat"] = henon_config["xx"].flatten()
    henon_config["y_flat"] = henon_config["yy"].flatten()
    henon_config["px_flat"] = np.zeros_like(henon_config["x_flat"])
    henon_config["py_flat"] = np.zeros_like(henon_config["x_flat"])

    henon_config["total_samples"] = henon_config["x_flat"].size

    henon_config["displacement"] = min(
        henon_config["dx"], henon_config["dy"]) * henon_config["displacement_scale"]

    displacement_table = np.array([sample_4d_displacement_on_a_sphere()
                                for _ in range(henon_config["total_samples"])])

    henon_config["x_random_displacement"] = henon_config["x_flat"] + \
        displacement_table[:, 0] * henon_config["displacement"]
    henon_config["y_random_displacement"] = henon_config["y_flat"] + \
        displacement_table[:, 1] * henon_config["displacement"]
    henon_config["px_random_displacement"] = henon_config["px_flat"] + \
        displacement_table[:, 2] * henon_config["displacement"]
    henon_config["py_random_displacement"] = henon_config["py_flat"] + \
        displacement_table[:, 3] * henon_config["displacement"]

    henon_config["x_displacement"] = henon_config["x_flat"] + \
        henon_config["displacement"]
    henon_config["y_displacement"] = henon_config["y_flat"] + \
        henon_config["displacement"]
    henon_config["px_displacement"] = henon_config["px_flat"] + \
        henon_config["displacement"]
    henon_config["py_displacement"] = henon_config["py_flat"] + \
        henon_config["displacement"]

    return henon_config

# get path of this script
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

@njit
def fit_3(x, a, k, c):
    return a / np.power(x, k) + c


def residual_3_fit(params, x, y):
    a = params["a"].value
    k = params["k"].value
    c = params["c"].value

    model = fit_3(x, a, k, c)
    y_clean = np.log10(y)
    mask = ~np.isnan(y_clean)
    return np.log10(model[mask]) - y_clean[mask]


def clean_data(x, y):
    x = x[~np.logical_or(np.logical_or(np.isnan(y), np.isinf(y)), y == 0)]
    y = y[~np.logical_or(np.logical_or(np.isnan(y), np.isinf(y)), y == 0)]
    return x, y


def fit(x, y, valid=True, kind="fit_fix_k", extra_log=False):
    # print(i)
    try:
        if extra_log:
            y = np.log10(y)
        
        x, y = clean_data(x, y)
        y = np.absolute(y)
        
        if len(x[x > 100]) < 2 or not valid:
            return "discarded"

        params = lmfit.Parameters()
        if kind == "fit_3":
            params.add("a", value=1, min=0)
            params.add("k", value=1)
            params.add("c", value=0)
            result = lmfit.minimize(
                residual_3_fit, params, args=(x, y))
        elif kind == "fit_fix_k":
            params.add("a", value=1, min=0)
            params.add("k", value=1, vary=False)
            params.add("c", value=0)
            result = lmfit.minimize(
                residual_3_fit, params, args=(x, y))
        elif kind == "fit_fix_a":
            params.add("a", value=1, vary=False)
            params.add("k", value=1)
            params.add("c", value=0)
            result = lmfit.minimize(
                residual_3_fit, params, args=(x, y))
        elif kind == "fit_fix_c":
            params.add("a", value=1, min=0)
            params.add("k", value=1)
            params.add("c", value=0, vary=False)
            result = lmfit.minimize(
                residual_3_fit, params, args=(x, y))
        else:
            raise ValueError(f"kind {kind} not recognized")
        return result
    except ValueError:
        # print(e)
        return "error"


def reject_outliers(data, m=10):
    return data[abs(data - np.mean(data)) < m * np.std(data)]


def clustering_2_points(data, clustering_method="KMeans", multiplier=30, ret_dx=False, basic=False):
    _, data = clean_data(np.ones_like(data), data)
    data = reject_outliers(data)
    if clustering_method == "GaussianMixtrue":
        labels = GaussianMixture(2).fit_predict(data.reshape(-1, 1))
    elif clustering_method == "KMeans":
        labels = KMeans(n_clusters=2).fit_predict(data.reshape(-1, 1))
    else:
        raise ValueError(f"clustering method {clustering_method} not recognized")
    max_1 = np.max(data[labels == 0])
    max_2 = np.max(data[labels == 1])
    min_1 = np.min(data[labels == 0])
    min_2 = np.min(data[labels == 1])
    if max_1 > max_2:
        thresh = (max_2 + min_1) / 2
    else:
        thresh = (max_1 + min_2) / 2
    
    if basic:
        return thresh

    X_max = np.max(data)
    X_min = np.min(data)
    X_samples, dX = np.linspace(X_min, X_max, 1000, retstep=True)

    data = data.reshape(-1, 1)
    kde = KernelDensity(kernel='gaussian', bandwidth=dX*multiplier).fit(data)

    def to_minimize(x):
        x = np.asarray(x).flatten()
        return np.exp(kde.score_samples(x.reshape(-1, 1)))

    values = to_minimize(X_samples)
    local_minima = X_samples[argrelextrema(values, np.less)]
    local_maxima = X_samples[argrelextrema(values, np.greater)]

    local_minima_left = local_minima[(local_minima - thresh) < 0]
    local_minima_right = local_minima[(local_minima - thresh) >= 0]
    local_maxima_left = local_maxima[(local_maxima - thresh) < 0]
    local_maxima_right = local_maxima[(local_maxima - thresh) >= 0]

    try:
        if len(local_minima_left) == 0 or len(local_minima_right) == 0:
            new_thresh = thresh
        else:
            if len(local_minima_left) == 0:
                new_thresh = local_minima_right[0]
            elif len(local_minima_right) == 0:
                new_thresh = local_minima_left[-1]
            else:
                if local_minima_left[-1] < local_maxima_left[-1]:
                    new_thresh = local_minima_right[0]
                else:
                    new_thresh = local_minima_left[-1]
    except Exception:
        new_thresh = thresh
    if not ret_dx:
        return new_thresh
    else:
        return new_thresh, dX * multiplier

class data_manager(object):
    DATA_DIR = os.path.join(SCRIPT_DIR, "../../data/")

    @staticmethod
    def parse_filename(filename):
        """Parse a filename like:
        'henon_ox_{args.omega_x}_oy_{args.omega_y}_modulation_{args.modulation_kind}_eps_{args.epsilon}_mu_{args.mu}_kmod_{args.kick_module}_o0_{args.omega_0}_disp_{args.displacement_kind}_data_{args.tracking}.hdf5'
        """
        filename_split = filename.split("_")
        return dict(
            omega_x = float(filename_split[2]),
            omega_y = float(filename_split[4]),
            modulation_kind = filename_split[6],
            epsilon = float(filename_split[8]),
            mu = float(filename_split[10]),
            kick_module = float(filename_split[12]),
            omega_0 = float(filename_split[14]),
            displacement_kind = filename_split[16],
            tracking = "_".join(filename_split[18:])[:-5]
        )

    @staticmethod
    def get_filename(omega_x, omega_y, modulation_kind, epsilon, mu, kick_module, omega_0, displacement_kind, tracking):
        return f"henon_ox_{omega_x}_oy_{omega_y}_modulation_{modulation_kind}_eps_{epsilon}_mu_{mu}_kmod_{kick_module}_o0_{omega_0}_disp_{displacement_kind}_data_{tracking}.hdf5"

    @staticmethod
    def ask_for_group(omega_x, omega_y, modulation_kind, epsilon, mu, kick_module, omega_0):
        return (omega_x, omega_y, modulation_kind, epsilon, mu, kick_module, omega_0)

    def refresh_files(self):
        self.file_list = os.listdir(self.DATA_DIR)
        self.file_list = list(filter(lambda x: x.startswith("henon_ox"), self.file_list))
        if len(self.file_list) == 0:
            print("No data files found.")
            self.groups = None
            return
        self.d_list = []
        self.groups = set()
        for filename in self.file_list:
            d = self.parse_filename(filename)
            t = tuple(d.values())
            # inser t in set
            self.groups.add(t)
            d["filename"] = filename
            self.d_list.append(d)

    def __init__(self, data_dir=None, create_files=True):
        if data_dir is not None:
            self.DATA_DIR = data_dir
            self.create_files = create_files
        
        self.henon_config = refresh_henon_config(HENON_BASE_CONFIG)
        self.refresh_files()
        
    def get_config(self):
        return self.henon_config

    def get_groups(self):
        return self.groups

    def get_times(self, times=None):
        if times is None:
            return self.henon_config["t_list"]
        
        # if times is not an iterable
        if not hasattr(times, "__iter__"):
            if times in self.henon_config["t_list"]:
                return np.array([times])
            else:
                print("We have not this time, I'll give you this one instead!")
                print(self.henon_config["t_list"][1])
                return np.array([self.henon_config["t_list"][1]])         
        else:
            to_return = []
            for t in times:
                if t in self.henon_config["t_list"]:
                    to_return.append(t)
                else:
                    print(f"We do not have {t}!")
            if len(to_return) == 0:
                return np.array([self.henon_config["t_list"][1]])
            else:
                return np.array(to_return)

    def get_file_from_group(self, group, displacement_kind, tracking, writing=False):
        print(f"Getting file for group {group} with displacement {displacement_kind} and tracking {tracking}!")
        
        filename = self.get_filename(group[0], group[1], group[2], group[3], group[4], group[5], group[6], displacement_kind, tracking)
        if filename in self.file_list:
            return h5py.File(os.path.join(self.DATA_DIR, filename), 
                mode='r' if not writing else 'r+')
        
        if not self.create_files:
            raise Exception("No file found and we are not allowed to create files!")

        print("Generating {} on the fly".format(group))
        hr.henon_run(
            omega_x=group[0], omega_y=group[1], modulation_kind=group[2], epsilon=group[3], mu=group[4], kick_module=group[5], omega_0=group[6],
            displacement_kind=displacement_kind, tracking=tracking, outdir=self.DATA_DIR, **self.henon_config
        )
        self.refresh_files()
        return self.get_file_from_group(group, displacement_kind, tracking, writing)

    def initial_radius(self):
        return np.sqrt(self.henon_config["x_flat"]**2 + self.henon_config["y_flat"]**2 + self.henon_config["px_flat"]**2 + self.henon_config["py_flat"]**2)

    def stability(self, group):
        f = self.get_file_from_group(group, "none", "track")
        data = f["steps"][:]
        f.close()
        return data

    def fast_lyapunov_indicator(self, group, times=None):
        f0 = self.get_file_from_group(group, "none", "lyapunov")
        times = self.get_times(times)
        data = {}
        for t in tqdm(times):
            data[t] = f0[f"lyapunov/{t}"][:] 
        data = pd.DataFrame(data=data)
        f0.close()
        return data

    def birkhoff_lyapunov_indicator(self, group, times=None):
        f0 = self.get_file_from_group(group, "none", "lyapunov_birkhoff")
        times = self.get_times(times)
        data = {}
        for t in tqdm(times):
            data[t] = f0[f"lyapunov/{t}"][:] 
        data = pd.DataFrame(data=data)
        f0.close()
        return data

    def orthogonal_lyapunov_indicator(self, group, times=None):
        f = self.get_file_from_group(group, "none", "orthogonal_lyapunov")
        times = self.get_times(times)
        data_max = {}
        data_avg = {}
        for i, t in tqdm(enumerate(times), total=len(times)):
            data_max[t] = np.max(f[f"lyapunov/{t}"][:], axis=1)
            data_avg[t] = np.mean(f[f"lyapunov/{t}"][:], axis=1)
        data_max = pd.DataFrame(data=data_max)
        data_avg = pd.DataFrame(data=data_avg)
        f.close()
        return data_max, data_avg
    
    def smallest_alignment_index(self, group, times=None):
        f0 = self.get_file_from_group(group, "none", "sali")
        times = self.get_times(times)
        data = {}
        for i, t in tqdm(enumerate(times), total=len(times)):
            data[t] = f0[f"sali/{t}"][:]
        data = pd.DataFrame(data=data)
        f0.close()
        return data

    def global_alignment_index(self, group, times=None):
        f0 = self.get_file_from_group(group, "none", "gali")
        times = self.get_times(times)
        data = {}
        for i, t in tqdm(enumerate(times), total=len(times)):
            data[t] = f0[f"gali/{t}"][:] 
        data = pd.DataFrame(data=data)
        f0.close()
        return data

    def reversibility_error(self, group, times=None):
        x0 = self.henon_config["x_flat"]
        y0 = self.henon_config["y_flat"]
        px0 = self.henon_config["px_flat"]
        py0 = self.henon_config["py_flat"]
        times = self.get_times(times)

        f = self.get_file_from_group(group, "none", "track_and_reverse")

        data = {}
        for t in tqdm(times):
            data[t] = di.reversibility_error(
                x0, px0, y0, py0,
                f[f"x/{t}"][:], f[f"px/{t}"][:], f[f"y/{t}"][:], f[f"py/{t}"][:]
            )
        data = pd.DataFrame(data=data)
        f.close()
        return data

    def megno(self, group, times=None):
        times = self.get_times(times)
        f = self.get_file_from_group(group, "none", "megno")
        data = {}
        for t in tqdm(times):
            data[t] = f[f"megno/{t}"][:]
        data = pd.DataFrame(data=data)
        f.close()
        return data

    def get_ground_truth(self, group, times=None, recompute=False):
        times = self.get_times(times)
        f = self.get_file_from_group(group, "none", "lyapunov")
        # check if the dataset "ground_truth" is available
        if "ground_truth" in f.keys() and not recompute:
            data = f["ground_truth"][:]
            f.close()
            return data
        
        f.close()
        f = self.get_file_from_group(group, "none", "lyapunov", writing=True) 

        print("Computing and saving ground truth")
        stability = f["steps"][:]
        stability = stability[:len(stability) // 2]
        
        lyapunov = self.fast_lyapunov_indicator(group, times)
        lyapunov = lyapunov[max(times)].to_numpy()

        threshold = clustering_2_points(np.log10(lyapunov))
        print("Saving ground truth")
        mask = np.asarray(stability==np.nanmax(stability), dtype=int)
        mask[stability <= 100] = -1
        mask[np.logical_and(
            np.log10(lyapunov) > threshold,
            np.logical_and(stability == np.nanmax(stability),
                           np.log10(lyapunov) < 0.7)
            )] = 2
        mask_df = f.require_dataset("ground_truth", shape=mask.shape, dtype=int, compression="gzip", shuffle=True)
        mask_df[:] = mask
        f.close()
        return mask
       
    def tangent_map(self, group, times=None):
        times = self.get_times(times)
        f = self.get_file_from_group(group, "none", "tangent_map")
        data = {}
        for t in tqdm(times):
            data[t] = np.log10(np.sqrt(f[f"tangent_map/{t}"][:])) / t
        data = pd.DataFrame(data=data)
        f.close()
        return data

    def fft_tunes(self, group, time_tresh=None):
        f = self.get_file_from_group(group, "none", "fft_tunes")
        data_x = {}
        data_y = {}
        for t_from in f["tune_x"].keys():
            data_x[t_from] = {}
            data_y[t_from] = {}
            for t_to in f["tune_x"][t_from].keys():
                data_x[t_from][t_to] = f[f"tune_x/{t_from}/{t_to}"][:]
                data_y[t_from][t_to] = f[f"tune_y/{t_from}/{t_to}"][:]
        f.close()
        return compute_tune_indicator_from_dict((data_x, data_y), time_tresh)

    def birkhoff_tunes(self, group, time_tresh=None):
        f = self.get_file_from_group(group, "none", "birkhoff_tunes")
        data_x = {}
        data_y = {}
        for t_from in f["tune_x"].keys():
            data_x[t_from] = {}
            data_y[t_from] = {}
            for t_to in f["tune_x"][t_from].keys():
                data_x[t_from][t_to] = f[f"tune_x/{t_from}/{t_to}"][:]
                data_y[t_from][t_to] = f[f"tune_y/{t_from}/{t_to}"][:]
        f.close()
        return compute_tune_indicator_from_dict((data_x, data_y), time_tresh)


def compute_tune_indicator_from_dict(double_dict, time_tresh=None):
    tx, ty = double_dict
    time_list = list(sorted(tx['0'].keys(), key=lambda x: int(x)))
    data = {}
    for t in time_list:
        if time_tresh is None or int(t) <= time_tresh:
            if t in tx:
                diff_x = np.power(tx[t][str(int(t)*2)] - tx['0'][t], 2)
                diff_y = np.power(ty[t][str(int(t)*2)] - ty['0'][t], 2)
                data[int(t)] = np.sqrt(diff_x + diff_y)
    data = pd.DataFrame(data=data)
    return data


@njit(parallel=True)
def convolution(data, kernel_size, mean_out, std_out):
    for i in prange(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isnan(data[i, j]):
                mean_out[i, j] = np.nan
                std_out[i, j] = np.nan
            else:
                mean_out[i, j] = np.nanmean(
                    data[i - kernel_size:i + kernel_size, j - kernel_size:j + kernel_size]
                )
                std_out[i, j] = np.nanstd(
                    data[i - kernel_size:i + kernel_size, j - kernel_size:j + kernel_size]
                )
    return mean_out, std_out


def apply_convolution(data, kernel_size, shape=(1000, 1000)):
    data = data.reshape(shape)
    mean_out = np.empty(shape)
    std_out = np.empty(shape)
    mean_out, std_out = convolution(data, kernel_size, mean_out, std_out)
    return mean_out.flatten(), std_out.flatten()


def apply_convolution_to_dataset(df, kernel_size, shape=(1000, 1000)):
    # iterate columns in df
    df_mean = {}
    df_std = {}
    for col in tqdm(df.columns):
        mean_out, std_out = apply_convolution(df[col].to_numpy(), kernel_size, shape)
        df_mean[col] = mean_out
        df_std[col] = std_out
    df_mean = pd.DataFrame(data=df_mean)
    df_std = pd.DataFrame(data=df_std)
    return df_mean, df_std


def classify_with_data(stab_data, dyn_data, dyn_thresh, stable_if_higher=False, naive_data=None, naive_thresh_min=100, naive_thresh_max=1000):
    bool_mask = np.logical_not(np.logical_or(
        np.isnan(stab_data), np.isnan(dyn_data)))
    stab_data = stab_data[bool_mask]
    dyn_data = dyn_data[bool_mask]
    data = stab_data  # >= stab_thresh
    guess = (dyn_data >= dyn_thresh) if stable_if_higher else (dyn_data <= dyn_thresh)

    if naive_data is None:
        naive_quota = 0
    else:
        naive_quota = np.count_nonzero((naive_data >= naive_thresh_min) & (naive_data <= naive_thresh_max))

    total = data.size + naive_quota
    tp = np.count_nonzero(data & guess)
    fp = np.count_nonzero(data & ~guess)
    fn = np.count_nonzero(~data & guess)
    tn = np.count_nonzero(~data & ~guess) + naive_quota

    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if tp + fp > 0 else np.nan
    recall = tp / (tp + fn) if tp + fn > 0 else np.nan
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else np.nan

    return dict(
        tp=tp, fp=fp, fn=fn, tn=tn, total=total, accuracy=accuracy,
        precision=precision, recall=recall, f1=f1, naive_quota=naive_quota
    )


def get_full_comparison(to_compare, bool_data, stable_if_higher, naive_data=None, naive_thresh_min=100, naive_thresh_max=1000):
    mask = np.logical_and(~np.isinf(to_compare), ~np.isnan(to_compare))
    to_compare = to_compare[mask]
    bool_data = bool_data[mask]
    data_range = (np.nanmin(to_compare), np.nanmax(to_compare))
    try:
        threshold = clustering_2_points(to_compare, clustering_method="KMeans", basic=True)
    except Exception:
        threshold = np.nanmedian(to_compare)
        
    samples = np.linspace(data_range[0], data_range[1], 100, dtype=float)

    confusion = []
    for s in samples:
        confusion.append(
            classify_with_data(
                bool_data,
                to_compare, s,
                stable_if_higher=stable_if_higher,
                naive_data=naive_data,
                naive_thresh_min=naive_thresh_min,
                naive_thresh_max=naive_thresh_max
            ))

    accuracy_all = np.array([c['accuracy'] for c in confusion])
    accuracy_best = np.nanmax(accuracy_all)
    accuracy_best_val = samples[np.argmax(accuracy_all)]
    
    f1_all = np.array([c['f1'] for c in confusion])
    f1_best = np.nanmax(f1_all)
    f1_best_val = samples[np.argmax(f1_all)]

    f1_best_accuracy = accuracy_all[np.argmax(f1_all)]
    accuracy_best_f1 = f1_all[np.argmax(accuracy_all)]

    confusion_threshold = classify_with_data(
        bool_data,
        to_compare, threshold,
        stable_if_higher=stable_if_higher,
        naive_data=naive_data,
        naive_thresh_min=naive_thresh_min,
        naive_thresh_max=naive_thresh_max
    )

    accuracy_threshold = confusion_threshold["accuracy"]
    f1_threshold = confusion_threshold["f1"]

    return dict(
        confusion=confusion, threshold=threshold,
        accuracy_all=accuracy_all, f1_all=f1_all,
        accuracy_threshold=accuracy_threshold, f1_threshold=f1_threshold,
        accuracy_best=accuracy_best, accuracy_best_val=accuracy_best_val,
        f1_best=f1_best, f1_best_val=f1_best_val,
        f1_best_accuracy=f1_best_accuracy, accuracy_best_f1=accuracy_best_f1
    )


def colorize(data, cmap="viridis", vmin=None, vmax=None, log10=False):
    if vmin is None:
        vmin = np.nanmin(data)
    if vmax is None:
        vmax = np.nanmax(data)
    if log10:
        data = np.log10(data)
        vmin = np.log10(vmin)
        vmax = np.log10(vmax)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    mapper = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    return mapper.to_rgba(data)


def downsample(data, step=10, initial_shape=(1000, 1000)):
    data = data.reshape(initial_shape)
    data = data[::step, ::step]
    return data.flatten()


def apply_downsample_to_dataset(df, step=10, initial_shape=(1000, 1000)):
    df_downsampled = {}
    for col in tqdm(df.columns):
        df_downsampled[col] = downsample(df[col].to_numpy(), step, initial_shape)
    df_downsampled = pd.DataFrame(data=df_downsampled)
    return df_downsampled

