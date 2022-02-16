from fileinput import filename
import numpy as np
import pandas as pd
import h5py
import os
import pickle
from tqdm import tqdm
from numba import njit, prange
from . import dynamic_indicators as di
from . import henon_run as hr
import matplotlib.pyplot as plt
import matplotlib

from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

RANDOM_SEED = 42
rs = RandomState(MT19937(SeedSequence(RANDOM_SEED)))

def sample_4d_displacement_on_a_sphere():
    n = rs.uniform(-1, 1, size=4)
    while n[0]**2 + n[1]**2 > 1 or n[2]**2 + n[3]**2 > 1:
        n = rs.uniform(-1, 1, size=4)
    fix = (1 - n[0]**2 - n[1]**2) / (n[2]**2 - n[3]**2)
    n[2] *= fix
    n[3] *= fix
    return n

HENON_BASE_CONFIG = {
    'name': 'henon_square_universal_config',
    'samples': 100,
    'random_seed': RANDOM_SEED,
    'displacement_scale': 1e-6,

    'x_extents': [0.0, 0.8],
    'y_extents': [0.0, 0.8],

    'low_tracking': 1000000,
    'tracking': 10000000,
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

    def __init__(self, data_dir=None):
        if data_dir is not None:
            self.DATA_DIR = data_dir
        
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

    def get_file_from_group(self, group, displacement_kind, tracking):
        print(f"Getting file for group {group} with displacement {displacement_kind} and tracking {tracking}!")
        
        filename = self.get_filename(group[0], group[1], group[2], group[3], group[4], group[5], group[6], displacement_kind, tracking)
        if filename in self.file_list:
            return h5py.File(os.path.join(self.DATA_DIR, filename), mode='r')
        
        print("Generating {} on the fly".format(group))
        hr.henon_run(
            group[0], group[1], group[2], group[3], group[4], group[5], group[6],
            displacement_kind, tracking, self.DATA_DIR, self.henon_config
        )
        self.refresh_files()
        return self.get_file_from_group(group, displacement_kind, tracking)

    def initial_radius(self):
        return np.sqrt(self.henon_config["x_flat"]**2 + self.henon_config["y_flat"]**2 + self.henon_config["px_flat"]**2 + self.henon_config["py_flat"]**2)

    def stability(self, group):
        f = self.get_file_from_group(group, "none", "track")
        data = f["steps"][:]
        f.close()
        return data

    def raw_displacement(self, group, displacement="random", times=None):
        f0 = self.get_file_from_group(group, "none", "step_track")
        f1 = self.get_file_from_group(group, displacement, "step_track")
        times = self.get_times(times)
        data = {}
        for t in tqdm(times):
            data[t] = di.raw_displacement(
                f0[f"x/{t}"][:], f0[f"px/{t}"][:], f0[f"y/{t}"][:], f0[f"py/{t}"][:],
                f1[f"x/{t}"][:], f1[f"px/{t}"][:], f1[f"y/{t}"][:], f1[f"py/{t}"][:],
            )
        data = pd.DataFrame(data=data)
        f0.close()
        f1.close()
        return data

    def fast_lyapunov_indicator(self, group, displacement="random", times=None):
        f0 = self.get_file_from_group(group, "none", "step_track")
        f1 = self.get_file_from_group(group, displacement, "step_track")
        times = self.get_times(times)
        data = {}
        for t in tqdm(times):
            data[t] = di.fast_lyapunov_indicator(
                f0[f"x/{t}"][:], f0[f"px/{t}"][:], f0[f"y/{t}"][:], f0[f"py/{t}"][:],
                f1[f"x/{t}"][:], f1[f"px/{t}"][:], f1[f"y/{t}"][:], f1[f"py/{t}"][:],
                self.henon_config["displacement"], t
            )
        data = pd.DataFrame(data=data)
        f0.close()
        f1.close()
        return data

    def invariant_lyapunov_indicator(self, group, times=None):
        f0 = self.get_file_from_group(group, "none", "step_track")
        f1 = self.get_file_from_group(group, "x", "step_track")
        f2 = self.get_file_from_group(group, "px", "step_track")
        f3 = self.get_file_from_group(group, "y", "step_track")
        f4 = self.get_file_from_group(group, "py", "step_track")
        times = self.get_times(times)
        data = {}
        for i, t in tqdm(enumerate(times), total=len(times)):
            data[t] = di.invariant_lyapunov_error(
                f0[f"x/{t}"][:], f0[f"px/{t}"][:], f0[f"y/{t}"][:], f0[f"py/{t}"][:],
                f1[f"x/{t}"][:], f1[f"px/{t}"][:], f1[f"y/{t}"][:], f1[f"py/{t}"][:],
                f2[f"x/{t}"][:], f2[f"px/{t}"][:], f2[f"y/{t}"][:], f2[f"py/{t}"][:],
                f3[f"x/{t}"][:], f3[f"px/{t}"][:], f3[f"y/{t}"][:], f3[f"py/{t}"][:],
                f4[f"x/{t}"][:], f4[f"px/{t}"][:], f4[f"y/{t}"][:], f4[f"py/{t}"][:],
                self.henon_config["displacement"], t
            )
        data = pd.DataFrame(data=data)
        f0.close()
        f1.close()
        f2.close()
        f3.close()
        f4.close()
        return data
    
    def smallest_alignment_index(self, group, times=None):
        f0 = self.get_file_from_group(group, "none", "step_track")
        f1 = self.get_file_from_group(group, "x", "step_track")
        f2 = self.get_file_from_group(group, "y", "step_track")
        times = self.get_times(times)
        data = {}
        for i, t in tqdm(enumerate(times), total=len(times)):
            data[t] = di.smallest_alignment_index(
                f0[f"x/{t}"][:], f0[f"px/{t}"][:], f0[f"y/{t}"][:], f0[f"py/{t}"][:],
                f1[f"x/{t}"][:], f1[f"px/{t}"][:], f1[f"y/{t}"][:], f1[f"py/{t}"][:],
                f2[f"x/{t}"][:], f2[f"px/{t}"][:], f2[f"y/{t}"][:], f2[f"py/{t}"][:]
            )
            if i > 0:
                data[t] = np.min(
                    [data[t], data[times[i-1]]],
                    axis=0
                )
        data = pd.DataFrame(data=data)
        f0.close()
        f1.close()
        f2.close()
        return data

    def global_alignment_index(self, group, times=None):
        f0 = self.get_file_from_group(group, "none", "step_track")
        f1 = self.get_file_from_group(group, "x", "step_track")
        f2 = self.get_file_from_group(group, "y", "step_track")
        f3 = self.get_file_from_group(group, "px", "step_track")
        f4 = self.get_file_from_group(group, "py", "step_track")
        times = self.get_times(times)
        data = {}
        for i, t in tqdm(enumerate(times), total=len(times)):
            data[t] = di.global_alignment_index(
                f0[f"x/{t}"][:], f0[f"px/{t}"][:], f0[f"y/{t}"][:], f0[f"py/{t}"][:],
                f1[f"x/{t}"][:], f1[f"px/{t}"][:], f1[f"y/{t}"][:], f1[f"py/{t}"][:],
                f2[f"x/{t}"][:], f2[f"px/{t}"][:], f2[f"y/{t}"][:], f2[f"py/{t}"][:],
                f3[f"x/{t}"][:], f3[f"px/{t}"][:], f3[f"y/{t}"][:], f3[f"py/{t}"][:],
                f4[f"x/{t}"][:], f4[f"px/{t}"][:], f4[f"y/{t}"][:], f4[f"py/{t}"][:]
            )
            if i > 0:
                data[t] = np.min(
                    [data[t], data[times[i-1]]],
                    axis=0
                )
        data = pd.DataFrame(data=data)
        f0.close()
        f1.close()
        f2.close()
        f3.close()
        f4.close()
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

    def better_lyapunov(self, group, times=None):
        times = self.get_times(times)
        f = self.get_file_from_group(group, "random", "true_displacement")
        data = {}
        for t in tqdm(times):
            data[t] = np.log10(f[f"displacement/{t}"][:]/self.henon_config["displacement"]) / t
        data = pd.DataFrame(data=data)
        f.close()
        return data

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


def classify_with_data(stab_data, dyn_data, stab_thresh, dyn_thresh, stable_if_higher=False):
    bool_mask = np.logical_not(np.logical_or(
        np.isnan(stab_data), np.isnan(dyn_data)))
    stab_data = stab_data[bool_mask]
    dyn_data = dyn_data[bool_mask]
    data = stab_data >= stab_thresh
    guess = (dyn_data >= dyn_thresh) if stable_if_higher else (
        dyn_data <= dyn_thresh)
    
    total = data.size
    tp = np.count_nonzero(data & guess)
    fp = np.count_nonzero(data & ~guess)
    fn = np.count_nonzero(~data & guess)
    tn = np.count_nonzero(~data & ~guess)

    accuracy = (tp + tn) / total
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return dict(
        tp=tp, fp=fp, fn=fn, tn=tn, total=total, accuracy=accuracy,
        precision=precision, recall=recall
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
