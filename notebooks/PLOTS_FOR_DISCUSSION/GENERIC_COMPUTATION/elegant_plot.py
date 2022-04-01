# %%
from turtle import title
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
import itertools
from sklearn.cluster import KMeans

from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches

# count number of threads
njobs = os.cpu_count()

# EPSILON_LIST = [0.0, 16.0, 64.0, 128.0]
EPSILON_LIST = [0.0, 16.0, 64.0]
MU_LIST = [0.0, 0.001, 0.01, 0.1]

OMEGA_LIST = [(0.310, 0.320, [0.0, 0.23], [0.0, 0.23]), (0.280, 0.310, [0.0, 0.5], [0.0, 0.5]), (0.168, 0.201, [0.0, 0.8], [0.0, 0.8])]
# OMEGA_LIST = [(0.310, 0.320, [0.0, 0.23], [0.0, 0.23])]

# OMEGA_X = 0.310
# OMEGA_Y = 0.320

X_EXTENTS = [0.0, 0.23]
Y_EXTENTS = [0.0, 0.23]

DATA_DIR = "./data"
SAMPLES = 100

T_GROUND_TRUTH = 100000000
T_NORMALIZATION = 1000
T_DYN_INDICATORS = 1000000

KERNEL_LIST = [3, 0, 5]

# for every combination of epsilon and mu

for (OMEGA_X, OMEGA_Y, X_EXTENTS, Y_EXTENTS), epsilon, mu, kernel in tqdm(list(itertools.product(OMEGA_LIST, EPSILON_LIST, MU_LIST, KERNEL_LIST))):
    
    # check if folder "figs/epsilon/mu" exists
    # if not, create it
    FIGPATH = f"figs/{OMEGA_X}_{OMEGA_Y}/{epsilon}_{mu}" if kernel == 0 else f"figs/{OMEGA_X}_{OMEGA_Y}/{epsilon}_{mu}/{kernel}"
    print(FIGPATH)
    if not os.path.exists(FIGPATH):
        os.makedirs(FIGPATH)

    group = (
        OMEGA_X,                  # omega_x
        OMEGA_Y,                  # omega_y
        "sps",                    # modulation_kind
        epsilon,                  # epsilon
        mu,                       # mu
        np.nan,                   # kick amplitude
        np.nan,                   # omega_0
    )

    group_random = tuple(
        [group[i] if i != 5 else 1e-10 for i in range(len(group))])

    try:

        # %%
        data = dm.data_manager(data_dir=DATA_DIR, create_files=False)

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
        with data.get_file_from_group(group, "none", "lyapunov") as f:
            stability = f["steps"][:]
        stability = stability[:len(stability)//2]
        stab_mask = ~np.isnan(full_lyapunov[T_GROUND_TRUTH].to_numpy())

        # %%
        data_minor = dm.data_manager(
            data_dir=os.path.join(DATA_DIR, "10_6"), create_files=False)

        data_minor.henon_config["samples"] = SAMPLES
        data_minor.henon_config["x_extents"] = X_EXTENTS
        data_minor.henon_config["y_extents"] = Y_EXTENTS

        data_minor.henon_config["t_base_2"] = np.array([], dtype=int)
        data_minor.henon_config["t_base"] = np.arange(
            T_NORMALIZATION, T_DYN_INDICATORS + T_NORMALIZATION, T_NORMALIZATION, dtype=int)

        data_minor.henon_config["t_base_10"] = np.array([], dtype=int)
        data_minor.henon_config["t_linear"] = np.array([], dtype=int)

        data_minor.henon_config = dm.refresh_henon_config(data_minor.henon_config)


        # %%
        times_minor = np.asarray(data_minor.get_times())


        # %%
        gali = data_minor.global_alignment_index(group)

        if kernel != 0:
            gali = dm.apply_convolution_to_dataset(gali, kernel, shape=(samples, samples))[0]

        # %%
        sali = data_minor.smallest_alignment_index(group)

        if kernel != 0:
            sali = dm.apply_convolution_to_dataset(sali, kernel, shape=(samples, samples))[0]

        # %%
        lyapunov = data_minor.fast_lyapunov_indicator(group)

        if kernel != 0:
            lyapunov = dm.apply_convolution_to_dataset(lyapunov, kernel, shape=(samples, samples))[0]

        # %%
        ortho_lyap_max, ortho_lyap_avg = data_minor.orthogonal_lyapunov_indicator(group)

        if kernel != 0:
            ortho_lyap_max = dm.apply_convolution_to_dataset(ortho_lyap_max, kernel, shape=(samples, samples))[0]
            ortho_lyap_avg = dm.apply_convolution_to_dataset(ortho_lyap_avg, kernel, shape=(samples, samples))[0]

        # %%
        reverse = data_minor.reversibility_error(group)

        if kernel != 0:
            reverse = dm.apply_convolution_to_dataset(reverse, kernel, shape=(samples, samples))[0]

        # %%
        #reverse_kick = data_minor.reversibility_error(group_random)

        # %%
        data_tune = dm.data_manager(
            data_dir=os.path.join(DATA_DIR, "10_6"), create_files=False)

        data_tune.henon_config["samples"] = SAMPLES
        data_tune.henon_config["x_extents"] = X_EXTENTS
        data_tune.henon_config["y_extents"] = Y_EXTENTS

        # find the closest power of 2 to T_DYN_INDICATORS
        t_tune_pow = np.ceil(np.log2(T_DYN_INDICATORS)) + 2
        t_tune_min = 4

        data_tune.henon_config["t_base_2"] = 2 ** np.arange(t_tune_min, t_tune_pow + 1, dtype=int)
        data_tune.henon_config["t_base"] = np.array([], dtype=int)
        data_tune.henon_config["t_base_10"] = np.array([], dtype=int)
        data_tune.henon_config["t_linear"] = np.array([], dtype=int)

        data_tune.henon_config = dm.refresh_henon_config(data_tune.henon_config)

        times_tune = np.asarray(data_tune.get_times())


        # %%
        tunes = data_tune.birkhoff_tunes(group)
    
    except Exception as e:
        print("Error in group {}".format(group))
        print(e)
        print("Moving forward...")
        continue

    if kernel != 0:
        tunes = dm.apply_convolution_to_dataset(
            tunes, kernel, shape=(samples, samples))[0]

    ############################################################################
    d = np.log10(full_lyapunov[100000000].to_numpy())
    
    # clean data from infs and nan
    d = d[np.isfinite(d)]
    d = d[np.logical_not(np.isnan(d))]

    labels = KMeans(n_clusters=2, random_state=42).fit_predict(d.reshape(-1, 1))
    max_1 = np.max(d[labels == 0])
    max_2 = np.max(d[labels == 1])
    min_1 = np.min(d[labels == 0])
    min_2 = np.min(d[labels == 1])
    if max_1 > max_2:
        thresh_1 = (max_2 + min_1) / 2
    else:
        thresh_1 = (max_1 + min_2) / 2

    d = np.log10(full_lyapunov[100000000].to_numpy())
    d = np.asarray(d <= thresh_1, dtype=float)
    d[~stab_mask] = np.nan

    ground_truth = d
    ############################################################################
    
    FIGPATH_COLORMAP = os.path.join(FIGPATH, "colormap")
    if not os.path.exists(FIGPATH_COLORMAP):
        os.makedirs(FIGPATH_COLORMAP)

    ############################################################################

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    mappable = axs[0].imshow(np.log10(stability.reshape(samples, samples)), cmap="viridis", extent=X_EXTENTS+Y_EXTENTS, origin="lower")
    fig.colorbar(mappable, ax=axs[0], label="log10(N turns)")
    axs[0].set_title("Stability")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")

    mappable = axs[1].imshow(np.log10(full_lyapunov[T_GROUND_TRUTH].to_numpy()).reshape(samples, samples), cmap="viridis", extent=X_EXTENTS+Y_EXTENTS, origin="lower")
    fig.colorbar(mappable, ax=axs[1], label="log10(Lyapunov at 10^8 steps)")
    axs[1].set_title("Lyapunov")
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("y")

    mappable = axs[2].imshow(ground_truth.reshape(samples, samples), cmap="viridis", extent=X_EXTENTS+Y_EXTENTS, origin="lower")
    axs[2].set_title("Ground truth")
    axs[2].set_xlabel("x")
    axs[2].set_ylabel("y")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGPATH_COLORMAP, "stable_colormaps.jpg"), dpi=300)
    
    # ############################################################################
    
    # FIGPATH_COLORMAP = os.path.join(FIGPATH, "colormap/gali")
    # if not os.path.exists(FIGPATH_COLORMAP):
    #     os.makedirs(FIGPATH_COLORMAP)

    # ############################################################################

    # for t in tqdm(times_minor):
    #     fig, axs = plt.subplots(1, 1)

    #     mappable = axs.imshow(np.log10(gali[t].to_numpy().reshape(samples, samples)), cmap="viridis", extent=X_EXTENTS+Y_EXTENTS, origin="lower")
    #     fig.colorbar(mappable, ax=axs, label="log10(N turns)")
    #     axs.set_title("GALI at t = {}".format(t))
    #     axs.set_xlabel("x")
    #     axs.set_ylabel("y")

    #     plt.tight_layout()
    #     plt.savefig(os.path.join(FIGPATH_COLORMAP, f"gali_{t}_colormaps.jpg"))

    # ############################################################################
    
    values_lyap = Parallel(n_jobs=njobs)(
    delayed(dm.get_full_comparison)(
        np.log10(lyapunov[t]).to_numpy(),
        ground_truth == 1,
        stable_if_higher=False,
        naive_data=stability,
        naive_thresh_min=100,
        naive_thresh_max=t
    ) for t in times_minor
    )

    th_best_lyap = np.asarray([v["accuracy_best_val"] for v in values_lyap])
    th_chosen_lyap = np.asarray([v["threshold"] for v in values_lyap])
    th_bestf1_lyap = np.asarray([v["f1_best_val"] for v in values_lyap])

    ac_best_lyap = np.asarray([v["accuracy_best"] for v in values_lyap])
    ac_chosen_lyap = np.asarray([v["accuracy_threshold"] for v in values_lyap])
    ac_bestf1_lyap = np.asarray([v["f1_best_accuracy"] for v in values_lyap])

    f1_best_lyap = np.asarray([v["accuracy_best_f1"] for v in values_lyap])
    f1_chosen_lyap = np.asarray([v["f1_threshold"] for v in values_lyap])
    f1_bestf1_lyap = np.asarray([v["f1_best"] for v in values_lyap])

    values_olyap_max = Parallel(n_jobs=njobs)(
        delayed(dm.get_full_comparison)(
            np.log10(ortho_lyap_max[t]).to_numpy(),
            ground_truth == 1,
            stable_if_higher=False,
            naive_data=stability,
            naive_thresh_min=100,
            naive_thresh_max=t
        ) for t in times_minor
    )

    th_best_olyap_max = np.asarray([v["accuracy_best_val"] for v in values_olyap_max])
    th_chosen_olyap_max = np.asarray([v["threshold"] for v in values_olyap_max])
    th_bestf1_olyap_max = np.asarray([v["f1_best_val"] for v in values_olyap_max])

    ac_best_olyap_max = np.asarray([v["accuracy_best"] for v in values_olyap_max])
    ac_chosen_olyap_max = np.asarray([v["accuracy_threshold"] for v in values_olyap_max])
    ac_bestf1_olyap_max = np.asarray([v["f1_best_accuracy"] for v in values_olyap_max])

    f1_best_olyap_max = np.asarray([v["accuracy_best_f1"] for v in values_olyap_max])
    f1_chosen_olyap_max = np.asarray([v["f1_threshold"] for v in values_olyap_max])
    f1_bestf1_olyap_max = np.asarray([v["f1_best"] for v in values_olyap_max])

    values_olyap_avg = Parallel(n_jobs=njobs)(
        delayed(dm.get_full_comparison)(
            np.log10(ortho_lyap_avg[t]).to_numpy(),
            ground_truth == 1,
            stable_if_higher=False,
            naive_data=stability,
            naive_thresh_min=100,
            naive_thresh_max=t
        ) for t in times_minor
    )

    th_best_olyap_avg = np.asarray([v["accuracy_best_val"] for v in values_olyap_avg])
    th_chosen_olyap_avg = np.asarray([v["threshold"] for v in values_olyap_avg])
    th_bestf1_olyap_avg = np.asarray([v["f1_best_val"] for v in values_olyap_avg])

    ac_best_olyap_avg = np.asarray([v["accuracy_best"] for v in values_olyap_avg])
    ac_chosen_olyap_avg = np.asarray([v["accuracy_threshold"] for v in values_olyap_avg])
    ac_bestf1_olyap_avg = np.asarray([v["f1_best_accuracy"] for v in values_olyap_avg])

    f1_best_olyap_avg = np.asarray([v["accuracy_best_f1"] for v in values_olyap_avg])
    f1_chosen_olyap_avg = np.asarray([v["f1_threshold"] for v in values_olyap_avg])
    f1_bestf1_olyap_avg = np.asarray([v["f1_best"] for v in values_olyap_avg])

    values_sali = Parallel(n_jobs=njobs)(
        delayed(dm.get_full_comparison)(
            np.log10(sali[t]).to_numpy(),
            ground_truth == 1,
            stable_if_higher=True,
            naive_data=stability,
            naive_thresh_min=100,
            naive_thresh_max=t
        ) for t in times_minor
    )

    th_best_sali = np.asarray([v["accuracy_best_val"] for v in values_sali])
    th_chosen_sali = np.asarray([v["threshold"] for v in values_sali])
    th_bestf1_sali = np.asarray([v["f1_best_val"] for v in values_sali])

    ac_best_sali = np.asarray([v["accuracy_best"] for v in values_sali])
    ac_chosen_sali = np.asarray([v["accuracy_threshold"] for v in values_sali])
    ac_bestf1_sali = np.asarray([v["f1_best_accuracy"] for v in values_sali])

    f1_best_sali = np.asarray([v["accuracy_best_f1"] for v in values_sali])
    f1_chosen_sali = np.asarray([v["f1_threshold"] for v in values_sali])
    f1_bestf1_sali = np.asarray([v["f1_best"] for v in values_sali])

    values_gali = Parallel(n_jobs=njobs)(
        delayed(dm.get_full_comparison)(
            np.log10(gali[t].to_numpy()),
            ground_truth == 1,
            stable_if_higher=True,
            naive_data=stability,
            naive_thresh_min=100,
            naive_thresh_max=t
        ) for t in times_minor
    )

    th_best_gali = np.asarray([v["accuracy_best_val"] for v in values_gali])
    th_chosen_gali = np.asarray([v["threshold"] for v in values_gali])
    th_bestf1_gali = np.asarray([v["f1_best_val"] for v in values_gali])

    ac_best_gali = np.asarray([v["accuracy_best"] for v in values_gali])
    ac_chosen_gali = np.asarray([v["accuracy_threshold"] for v in values_gali])
    ac_bestf1_gali = np.asarray([v["f1_best_accuracy"] for v in values_gali])

    f1_best_gali = np.asarray([v["accuracy_best_f1"] for v in values_gali])
    f1_chosen_gali = np.asarray([v["f1_threshold"] for v in values_gali])
    f1_bestf1_gali = np.asarray([v["f1_best"] for v in values_gali])

    values_reverse = Parallel(n_jobs=njobs)(
        delayed(dm.get_full_comparison)(
            np.log10(reverse[t]).to_numpy(),
            ground_truth == 1,
            stable_if_higher=False,
            naive_data=stability,
            naive_thresh_min=100,
            naive_thresh_max=t
        ) for t in times_minor
    )

    th_best_reverse = np.asarray([v["accuracy_best_val"] for v in values_reverse])
    th_chosen_reverse = np.asarray([v["threshold"] for v in values_reverse])
    th_bestf1_reverse = np.asarray([v["f1_best_val"] for v in values_reverse])

    ac_best_reverse = np.asarray([v["accuracy_best"] for v in values_reverse])
    ac_chosen_reverse = np.asarray([v["accuracy_threshold"] for v in values_reverse])
    ac_bestf1_reverse = np.asarray([v["f1_best_accuracy"] for v in values_reverse])

    f1_best_reverse = np.asarray([v["accuracy_best_f1"] for v in values_reverse])
    f1_chosen_reverse = np.asarray([v["f1_threshold"] for v in values_reverse])
    f1_bestf1_reverse = np.asarray([v["f1_best"] for v in values_reverse])

    values_tunes = Parallel(n_jobs=njobs)(
        delayed(dm.get_full_comparison)(
            np.log10(tunes[t]).to_numpy(),
            ground_truth == 1,
            stable_if_higher=False,
            naive_data=stability,
            naive_thresh_min=100,
            naive_thresh_max=t
        ) for t in times_tune[:-2]
    )

    th_best_tunes = np.asarray([v["accuracy_best_val"] for v in values_tunes])
    th_chosen_tunes = np.asarray([v["threshold"] for v in values_tunes])
    th_bestf1_tunes = np.asarray([v["f1_best_val"] for v in values_tunes])

    ac_best_tunes = np.asarray([v["accuracy_best"] for v in values_tunes])
    ac_chosen_tunes = np.asarray([v["accuracy_threshold"] for v in values_tunes])
    ac_bestf1_tunes = np.asarray([v["f1_best_accuracy"] for v in values_tunes])

    f1_best_tunes = np.asarray([v["accuracy_best_f1"] for v in values_tunes])
    f1_chosen_tunes = np.asarray([v["f1_threshold"] for v in values_tunes])
    f1_bestf1_tunes = np.asarray([v["f1_best"] for v in values_tunes])

    ############################################################################
    
    fig, ax = plt.subplots(1, 1)

    ax.plot(times_minor, ac_best_gali, label="GALI")
    ax.plot(times_minor, ac_best_sali, label="SALI")
    ax.plot(times_minor, ac_best_reverse, label="Reverse")
    ax.plot(times_minor, ac_best_lyap, label="FLI")
    ax.plot(times_minor, ac_best_olyap_max, label="OFLI max")
    ax.plot(times_minor, ac_best_olyap_avg, label="OFLI avg")
    ax.plot(times_tune[5:-2], ac_best_tunes[5:], label="Tune")

    ax.legend(ncol=2, fontsize="small")
    ax.set_xscale("log")

    ax.set_xlim(1e3, 1e6)
    ax.set_xlabel("$N$ turns")
    ax.set_ylabel("Accuracy")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGPATH, "a_posteriori" + ".jpg"), dpi=300)

    ############################################################################
    
    fig, ax = plt.subplots(1, 1)

    ax.plot(times_minor, ac_chosen_gali, label="GALI")
    ax.plot(times_minor, ac_chosen_sali, label="SALI")
    ax.plot(times_minor, ac_chosen_reverse, label="Reverse")
    ax.plot(times_minor, ac_chosen_lyap, label="FLI")
    ax.plot(times_minor, ac_chosen_olyap_max, label="OFLI max")
    ax.plot(times_minor, ac_chosen_olyap_avg, label="OFLI avg")
    ax.plot(times_tune[5:-2], ac_chosen_tunes[5:], label="Tune")

    ax.legend(ncol=2, fontsize="small")
    ax.set_xscale("log")

    ax.set_xlim(1e3, 1e6)
    ax.set_xlabel("$N$ turns")
    ax.set_ylabel("Accuracy")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGPATH, "a_priori" + ".jpg"), dpi=300)

    ############################################################################

    fig, ax = plt.subplots(1, 1)

    ax.plot([], [], color="grey", label="Accuracy a priori")
    ax.plot([], [], color="grey", linestyle="--", label="F1 a priori")
    ax.plot([], [], color="grey", linestyle="dotted", label="Accuracy a posteriori")
    ax.plot([], [], color="white", label=" ")

    ax.plot(times_minor, ac_chosen_gali, label="GALI", c="C0")
    ax.plot(times_minor, ac_chosen_sali, label="SALI", c="C1")
    ax.plot(times_minor, ac_chosen_reverse, label="Reverse", c="C2")
    ax.plot(times_minor, ac_chosen_lyap, label="FLI", c="C3")
    ax.plot(times_minor, ac_chosen_olyap_max, label="OFLI max", c="C4")
    ax.plot(times_minor, ac_chosen_olyap_avg, label="OFLI avg", c="C5")
    ax.plot(times_tune[5:-2], ac_chosen_tunes[5:], label="Tune", c="C6")

    ax.plot(times_minor, f1_chosen_gali, linestyle="--", c="C0")
    ax.plot(times_minor, f1_chosen_sali, linestyle="--", c="C1")
    ax.plot(times_minor, f1_chosen_reverse, linestyle="--", c="C2")
    ax.plot(times_minor, f1_chosen_lyap, linestyle="--", c="C3")
    ax.plot(times_minor, f1_chosen_olyap_max, linestyle="--", c="C4")
    ax.plot(times_minor, f1_chosen_olyap_avg, linestyle="--", c="C5")
    ax.plot(times_tune[5:-2], f1_chosen_tunes[5:], linestyle="--", c="C6")

    ax.plot(times_minor, ac_best_gali, linestyle="dotted", c="C0")
    ax.plot(times_minor, ac_best_sali, linestyle="dotted", c="C1")
    ax.plot(times_minor, ac_best_reverse, linestyle="dotted", c="C2")
    ax.plot(times_minor, ac_best_lyap, linestyle="dotted", c="C3")
    ax.plot(times_minor, ac_best_olyap_max, linestyle="dotted", c="C4")
    ax.plot(times_minor, ac_best_olyap_avg, linestyle="dotted", c="C5")
    ax.plot(times_tune[5:-2], ac_best_tunes[5:], linestyle="dotted", c="C6")

    ax.legend(ncol=3, fontsize="small")
    ax.set_xscale("log")

    ax.set_xlim(1e3, 1e6)
    ax.set_xlabel("$N$ turns")
    ax.set_ylabel("Accuracy")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGPATH, "performance_plot" + ".jpg"), dpi=300)

    ############################################################################

    ZOOM_LEVELS = [0.5, 0.75, 0.9, 0.95, 0.99]

    ac_chosen_list = [ac_chosen_gali, ac_chosen_sali, ac_chosen_reverse, ac_chosen_lyap, ac_chosen_olyap_max, ac_chosen_olyap_avg]
    f1_chosen_list = [f1_chosen_gali, f1_chosen_sali, f1_chosen_reverse, f1_chosen_lyap, f1_chosen_olyap_max, f1_chosen_olyap_avg]
    ac_best_list = [ac_best_gali, ac_best_sali, ac_best_reverse, ac_best_lyap, ac_best_olyap_max, ac_best_olyap_avg]
    label_list = ["GALI", "SALI", "Reverse", "FLI", "OFLI max", "OFLI avg"]
    color_list = ["C0", "C1", "C2", "C3", "C4", "C5"]

    for zoom in ZOOM_LEVELS:
        for ac_chosen, f1_chosen, ac_best, label, color in zip(ac_chosen_list, f1_chosen_list, ac_best_list, label_list, color_list):
            fig, ax = plt.subplots(1, 1)

            ax.plot(times_minor, ac_chosen, label="Accuracy a priori", c=color)
            ax.plot(times_minor, f1_chosen, linestyle="dashed", label="F1 a priori", c=color)
            ax.plot(times_minor, ac_best, linestyle="dotted", label="Accuracy a posteriori", c=color)

            ax.legend(ncol=1, fontsize="small", title="Dynamic Indicator: " + label)
            ax.set_xscale("log")

            ax.set_xlim(1e3, 1e6)
            ax.set_ylim(zoom, 1.0)
            ax.set_xlabel("$N$ turns")
            ax.set_ylabel("Accuracy / F1 value")

            plt.tight_layout()
            plt.savefig(os.path.join(FIGPATH, "performance_" + label + f"_z_{zoom}.jpg"), dpi=300)


        fig, ax = plt.subplots(1, 1)

        ax.plot(times_tune[5:-2], ac_chosen_tunes[5:], label="Accuracy a priori", c="C6")
        ax.plot(times_tune[5:-2], f1_chosen_tunes[5:], linestyle="dashed", label="F1 a priori", c="C6")
        ax.plot(times_tune[5:-2], ac_best_tunes[5:], linestyle="dotted", label="Accuracy a posteriori", c="C6")

        ax.legend(ncol=1, fontsize="small", title="Dynamic Indicator: Tune")
        ax.set_xscale("log")

        ax.set_xlim(1e3, 1e6)
        ax.set_ylim(zoom, 1.0)
        ax.set_xlabel("$N$ turns")
        ax.set_ylabel("Accuracy / F1 value")

        plt.tight_layout()
        plt.savefig(os.path.join(FIGPATH, f"performance_tune_z_{zoom}.jpg"), dpi=300)
    ############################################################################

    # %% [markdown]
    # # PLOT FULL LYAPUNOV...

    # %%
    TIMES = times
    DATA = full_lyapunov

    # %%
    def cover_extreme_outliers(data, m=10):
        """
        Cover extreme outliers in data.
        """
        data_mean = np.nanmean(data)
        data_std = np.nanstd(data)
        data_min = data_mean - m * data_std
        data_max = data_mean + m * data_std
        data_mask = np.logical_and(data >= data_min, data <= data_max)
        data[~data_mask] = np.nan
        return data

    # %%
    val_min = np.nan
    val_max = np.nan
    for k in DATA.keys():
        d = np.log10(DATA[k].to_numpy())[stab_mask]
        d = cover_extreme_outliers(d)
        val_min = np.nanmin([val_min, np.nanmin(d)])
        val_max = np.nanmax([val_max, np.nanmax(d)])
    print(val_min, val_max)

    # %%
    count, bins = np.histogram(np.log10(DATA[1000000].to_numpy()),
                bins=50, range=(val_min, val_max))

    bin_centers = (bins[:-1] + bins[1:]) / 2

    # %%
    nbins = 50
    count_map = np.zeros((len(TIMES), nbins))

    for i, k in enumerate(DATA.keys()):
        d = np.log10(DATA[k].to_numpy())[stab_mask]
        count, bins = np.histogram(d, bins=50, range=(val_min, val_max), density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        count_map[i, :] = count

    count_map[count_map==0] = np.nan

    # %%
    fig = plt.figure(figsize=(10, 8))
    # fig, axs = plt.subplot_mosaic(
    #     [["cbar","cmap","cmap","cmap","cmap", "hist1","hist1","hist1"],
    #      ["cbar","cmap","cmap","cmap","cmap", "hist2","hist2","hist2"],
    #      ["cbar", "cmap", "cmap", "cmap","cmap", "hist3", "hist3", "hist3"]],
    #     figsize=(10, 10))

    gs = GridSpec(3, 3, figure=fig, width_ratios=[0.2, 1, 1])
    axs = {}
    axs["cmap"] = fig.add_subplot(gs[:, 1])
    axs["cbar"] = fig.add_subplot(gs[:, 0])
    axs["hist1"] = fig.add_subplot(gs[0, 2])
    axs["hist2"] = fig.add_subplot(gs[1, 2])
    axs["hist3"] = fig.add_subplot(gs[2, 2])

    mappable = axs["cmap"].imshow(np.log10(count_map), aspect="auto", interpolation="nearest", extent=(val_min, val_max, 0, len(TIMES)), origin="lower")


    axs["cmap"].set_xlabel("$\\log_{{10}}(FLI)$")
    axs["cmap"].set_ylabel("$N$ turns")
    # set the ticks on the y axis to be the times
    plt.sca(axs["cmap"])
    plt.yticks(np.arange(len(TIMES))+0.5, [f"$10^{int(np.log10(t))}$" for t in TIMES])

    axs["hist1"].hist(np.log10(DATA[TIMES[-1]].to_numpy()), bins=50, range=(val_min, val_max), density=True)
    axs["hist2"].hist(np.log10(DATA[TIMES[len(TIMES)//2]].to_numpy()), bins=50, range=(val_min, val_max), density=True)
    axs["hist3"].hist(np.log10(DATA[TIMES[0]].to_numpy()), bins=50, range=(val_min, val_max), density=True)

    ############################################################################
    ############################################################################

    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture

    d = np.log10(full_lyapunov[100000000].to_numpy())
    # clean data from infs and nan
    d = d[np.isfinite(d)]
    d = d[np.logical_not(np.isnan(d))]

    labels = KMeans(n_clusters=2, random_state=42).fit_predict(d.reshape(-1, 1))
    max_1 = np.max(d[labels == 0])
    max_2 = np.max(d[labels == 1])
    min_1 = np.min(d[labels == 0])
    min_2 = np.min(d[labels == 1])
    if max_1 > max_2:
        thresh_1 = (max_2 + min_1) / 2
    else:
        thresh_1 = (max_1 + min_2) / 2

    axs["hist1"].axvline(thresh_1, color="r", linestyle="--", label="KMeans based\nground truth")
    axs["hist1"].legend()

    ############################################################################
    ############################################################################

    axs["hist1"].set_ylabel("Frequency")
    axs["hist2"].set_ylabel("Frequency")
    axs["hist3"].set_ylabel("Frequency")
    axs["hist3"].set_xlabel("$\\log_{{10}}(FLI)$")

    # move ylabels to the right
    axs["hist1"].yaxis.tick_right()
    axs["hist2"].yaxis.tick_right()
    axs["hist3"].yaxis.tick_right()
    # move y title to the right
    axs["hist1"].yaxis.set_label_position("right")
    axs["hist2"].yaxis.set_label_position("right")
    axs["hist3"].yaxis.set_label_position("right")

    # set titles for the histos
    axs["hist1"].set_title(f"$10^{int(np.log10(TIMES[-1]))}$ turns")
    axs["hist2"].set_title(f"$10^{int(np.log10(TIMES[len(TIMES)//2]))}$ turns")
    axs["hist3"].set_title(f"$10^{int(np.log10(TIMES[0]))}$ turns")

    d_rect_x = (val_max - val_min) * 0.005
    d_rect_y = 0.01
    rect = patches.Rectangle((val_min+d_rect_x, len(TIMES)-1+d_rect_y), val_max -
                            val_min-d_rect_x*2, 1-d_rect_y*2, linewidth=3, edgecolor='r', facecolor='none')
    axs["cmap"].add_patch(rect)

    rect = patches.Rectangle((val_min+d_rect_x, 0+d_rect_y), val_max - val_min-d_rect_x*2, 1-d_rect_y*2, linewidth=3, edgecolor='r', facecolor='none')
    axs["cmap"].add_patch(rect)


    rect = patches.Rectangle((val_min+d_rect_x, len(TIMES)//2+d_rect_y), val_max - val_min-d_rect_x*2, 1-d_rect_y*2, linewidth=3, edgecolor='r', facecolor='none')
    axs["cmap"].add_patch(rect)


    # clear completely axs["cbar"]
    fig.delaxes(axs["cbar"])

    # create colorbar with height equal to axs["cmap"]
    fig.colorbar(mappable, ax=axs["cbar"],
                location="left", label="Frequency $[\\log_{10}]$", aspect=20)
    plt.tight_layout()

    plt.savefig(os.path.join(FIGPATH, "ground_truth_hist.jpg"), dpi=300)

    # %% [markdown]
    # # TUNE PLOT
    TIMES = times_tune
    DATA = tunes
    TH_BEST = th_best_tunes
    TH_CHOSEN = th_chosen_tunes
    TH_F1 = th_bestf1_tunes

    picked_times = [
        1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072
    ]
    corresponding_indexes = [
        np.where(TIMES == t)[0][0] for t in picked_times
    ]

    labels = [f"${t}$" for t in picked_times]

    # %%
    def cover_extreme_outliers(data, m=10):
        """
        Cover extreme outliers in data.
        """
        data_mean = np.nanmean(data)
        data_std = np.nanstd(data)
        data_min = data_mean - m * data_std
        data_max = data_mean + m * data_std
        data_mask = np.logical_and(data >= data_min, data <= data_max)
        data[~data_mask] = np.nan
        return data

    # %%
    val_min = np.nan
    val_max = np.nan
    for k in picked_times:
        d = np.log10(DATA[k].to_numpy())[stab_mask]
        d[np.isinf(d)] = np.nan
        d = cover_extreme_outliers(d)
        val_min = np.nanmin([val_min, np.nanmin(d)])
        val_max = np.nanmax([val_max, np.nanmax(d)])
    print(val_min, val_max)

    # %%
    count, bins = np.histogram(np.log10(DATA[picked_times[-1]].to_numpy()),
                                bins=50, range=(val_min, val_max))

    bin_centers = (bins[:-1] + bins[1:]) / 2

    # %%
    nbins = 50
    count_map = np.zeros((len(picked_times), nbins))

    for i, k in enumerate(picked_times):
        d = np.log10(DATA[k].to_numpy())[stab_mask]
        count, bins = np.histogram(
            d, bins=50, range=(val_min, val_max), density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        count_map[i, :] = count

    count_map[count_map == 0] = np.nan

    # %%
    fig = plt.figure(figsize=(10, 8))
    # fig, axs = plt.subplot_mosaic(
    #     [["cbar","cmap","cmap","cmap","cmap", "hist1","hist1","hist1"],
    #      ["cbar","cmap","cmap","cmap","cmap", "hist2","hist2","hist2"],
    #      ["cbar", "cmap", "cmap", "cmap","cmap", "hist3", "hist3", "hist3"]],
    #     figsize=(10, 10))

    gs = GridSpec(3, 3, figure=fig, width_ratios=[0.2, 1, 1])
    axs = {}
    axs["cmap"] = fig.add_subplot(gs[:, 1])
    axs["cbar"] = fig.add_subplot(gs[:, 0])
    axs["hist1"] = fig.add_subplot(gs[0, 2])
    axs["hist2"] = fig.add_subplot(gs[1, 2])
    axs["hist3"] = fig.add_subplot(gs[2, 2])

    mappable = axs["cmap"].imshow(np.log10(count_map), aspect="auto", interpolation="nearest", extent=(
        val_min, val_max, 0, len(picked_times)), origin="lower")

    axs["cmap"].set_xlabel("$\\log_{{10}}($Tune diff$)$")
    axs["cmap"].set_ylabel("$N$ turns")
    # set the ticks on the y axis to be the times
    plt.sca(axs["cmap"])
    plt.yticks(np.arange(len(picked_times))+0.5, labels)

    axs["hist1"].hist(np.log10(DATA[picked_times[-1]].to_numpy()),
                        bins=50, range=(val_min, val_max), density=True)
    axs["hist2"].hist(np.log10(DATA[picked_times[len(
        picked_times)//2]].to_numpy()), bins=50, range=(val_min, val_max), density=True)
    axs["hist3"].hist(np.log10(DATA[picked_times[0]].to_numpy()),
                        bins=50, range=(val_min, val_max), density=True)

    axs["hist1"].axvline(
        TH_BEST[corresponding_indexes[-1]], c="aqua", linestyle="--", label="A posteriori threshold")
    axs["hist1"].axvline(
        TH_CHOSEN[corresponding_indexes[-1]], c="r", linestyle="--", label="Basic KMeans threshold")
    axs["hist1"].axvline(
        TH_F1[corresponding_indexes[-1]], c="grey", linestyle="--", label="Best F1 threshold")

    axs["hist2"].axvline(
        TH_BEST[corresponding_indexes[len(picked_times)//2]], c="aqua", linestyle="--", label="A posteriori threshold")
    axs["hist2"].axvline(
        TH_CHOSEN[corresponding_indexes[len(picked_times)//2]], c="r", linestyle="--", label="Basic KMeans threshold")
    axs["hist2"].axvline(
        TH_F1[corresponding_indexes[len(picked_times)//2]], c="grey", linestyle="--", label="Best F1 threshold")

    axs["hist3"].axvline(
        TH_BEST[corresponding_indexes[0]], c="aqua", linestyle="--", label="A posteriori threshold")
    axs["hist3"].axvline(
        TH_CHOSEN[corresponding_indexes[0]], c="r", linestyle="--", label="Basic KMeans threshold")
    axs["hist3"].axvline(
        TH_F1[corresponding_indexes[0]], c="grey", linestyle="--", label="Best F1 threshold")

    axs["hist1"].legend(fontsize="small")
    axs["hist2"].legend(fontsize="small")
    axs["hist3"].legend(fontsize="small")

    axs["hist1"].set_ylabel("Frequency")
    axs["hist2"].set_ylabel("Frequency")
    axs["hist3"].set_ylabel("Frequency")
    axs["hist3"].set_xlabel("$\\log_{{10}}($Tune diff$)$")
    # move ylabels to the right
    axs["hist1"].yaxis.tick_right()
    axs["hist2"].yaxis.tick_right()
    axs["hist3"].yaxis.tick_right()
    # move y title to the right
    axs["hist1"].yaxis.set_label_position("right")
    axs["hist2"].yaxis.set_label_position("right")
    axs["hist3"].yaxis.set_label_position("right")

    # set titles for the histos
    axs["hist1"].set_title(f"{labels[-1]} turns")
    axs["hist2"].set_title(f"{labels[len(picked_times)//2]} turns")
    axs["hist3"].set_title(f"{labels[0]} turns")

    for pos, idx in enumerate(corresponding_indexes):
        axs["cmap"].plot([TH_BEST[idx], TH_BEST[idx]], [pos, pos+1], c="aqua", linestyle="--")
        axs["cmap"].plot([TH_CHOSEN[idx], TH_CHOSEN[idx]], [pos, pos+1], c="r", linestyle="--")
        axs["cmap"].plot([TH_F1[idx], TH_F1[idx]], [pos, pos+1], c="grey", linestyle="--")

    d_rect_x = (val_max - val_min) * 0.005
    d_rect_y = 0.01
    rect = patches.Rectangle((val_min+d_rect_x, len(picked_times)-1+d_rect_y), val_max -
                                val_min-d_rect_x*2, 1-d_rect_y*2, linewidth=3, edgecolor='r', facecolor='none')
    axs["cmap"].add_patch(rect)

    rect = patches.Rectangle((val_min+d_rect_x, 0+d_rect_y), val_max - val_min- \
                                d_rect_x*2, 1-d_rect_y*2, linewidth=3, edgecolor='r', facecolor='none')
    axs["cmap"].add_patch(rect)

    rect = patches.Rectangle((val_min+d_rect_x, len(picked_times)//2+d_rect_y), val_max -
                                val_min-d_rect_x*2, 1-d_rect_y*2, linewidth=3, edgecolor='r', facecolor='none')
    axs["cmap"].add_patch(rect)

    # clear completely axs["cbar"]
    fig.delaxes(axs["cbar"])

    # create colorbar with height equal to axs["cmap"]
    fig.colorbar(mappable, ax=axs["cbar"],
                    location="left", label="Frequency $[\\log_{10}]$", aspect=20)
    plt.tight_layout()

    plt.savefig(os.path.join(FIGPATH, "tune" + ".jpg"), dpi=300)

    # %% [markdown]
    # # THE PLOT

    # %%
    TIMES = times_minor
    
    DATA_LIST = [lyapunov, gali, sali, ortho_lyap_avg, ortho_lyap_max, reverse]
    TITLE_LIST = ["Lyapunov", "GALI", "SALI", "Ortho Lyapunov avg", "Ortho Lyapunov max", "Reverse"]
    LABEL_LIST = ["$\\log_{10}(FLI)$", "$\\log_{10}(GALI)$", "$\\log_{10}(SALI)$", "$\\log_{10}(OFLI avg)$", "$\\log_{10}(OFLI max)$", "$\\log_{10}(REM)$"]

    TH_BEST_LIST = [th_best_lyap, th_best_gali, th_best_sali, th_best_olyap_avg, th_best_olyap_max, th_best_reverse]
    TH_CHOSEN_LIST = [th_chosen_lyap, th_chosen_gali, th_chosen_sali, th_chosen_olyap_avg, th_chosen_olyap_max, th_chosen_reverse]
    TH_F1_BEST_LIST = [th_bestf1_lyap, th_bestf1_gali, th_bestf1_sali, th_bestf1_olyap_avg, th_bestf1_olyap_max, th_bestf1_reverse]

    for DATA, TITLE, LABEL, TH_BEST, TH_CHOSEN, TH_F1 in tqdm(zip(DATA_LIST, TITLE_LIST, LABEL_LIST, TH_BEST_LIST, TH_CHOSEN_LIST, TH_F1_BEST_LIST)):
        print(TITLE)
        # %%
        picked_times = [
            1000, 3000,
            10000, 30000,
            100000, 300000,
            1000000
        ]
        corresponding_indexes = [
            np.where(TIMES == t)[0][0] for t in picked_times
        ]

        labels = [
            "$1\\times 10^3$", "$3\\times 10^3$",
            "$1\\times 10^4$", "$3\\times 10^4$",
            "$1\\times 10^5$", "$3\\times 10^5$",
            "$1\\times 10^6$"
        ]

        # %%
        def cover_extreme_outliers(data, m=10):
            """
            Cover extreme outliers in data.
            """
            data_mean = np.nanmean(data)
            data_std = np.nanstd(data)
            data_min = data_mean - m * data_std
            data_max = data_mean + m * data_std
            data_mask = np.logical_and(data >= data_min, data <= data_max)
            data[~data_mask] = np.nan
            return data

        # %%
        val_min = np.nan
        val_max = np.nan
        for k in picked_times:
            d = np.log10(DATA[k].to_numpy())[stab_mask]
            d[np.isinf(d)] = np.nan
            d = cover_extreme_outliers(d)
            val_min = np.nanmin([val_min, np.nanmin(d)])
            val_max = np.nanmax([val_max, np.nanmax(d)])
        print(val_min, val_max)


        # %%
        count, bins = np.histogram(np.log10(DATA[1000000].to_numpy()),
                    bins=50, range=(val_min, val_max))

        bin_centers = (bins[:-1] + bins[1:]) / 2

        # %%
        nbins = 50
        count_map = np.zeros((len(picked_times), nbins))

        for i, k in enumerate(picked_times):
            d = np.log10(DATA[k].to_numpy())[stab_mask]
            count, bins = np.histogram(d, bins=50, range=(val_min, val_max), density=True)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            count_map[i, :] = count

        count_map[count_map==0] = np.nan


        # %%
        fig = plt.figure(figsize=(10, 8))
        # fig, axs = plt.subplot_mosaic(
        #     [["cbar","cmap","cmap","cmap","cmap", "hist1","hist1","hist1"],
        #      ["cbar","cmap","cmap","cmap","cmap", "hist2","hist2","hist2"],
        #      ["cbar", "cmap", "cmap", "cmap","cmap", "hist3", "hist3", "hist3"]],
        #     figsize=(10, 10))

        gs = GridSpec(3, 3, figure=fig, width_ratios=[0.2, 1, 1])
        axs = {}
        axs["cmap"] = fig.add_subplot(gs[:, 1])
        axs["cbar"] = fig.add_subplot(gs[:, 0])
        axs["hist1"] = fig.add_subplot(gs[0, 2])
        axs["hist2"] = fig.add_subplot(gs[1, 2])
        axs["hist3"] = fig.add_subplot(gs[2, 2])

        mappable = axs["cmap"].imshow(np.log10(count_map), aspect="auto", interpolation="nearest", extent=(val_min, val_max, 0, len(picked_times)), origin="lower")

        axs["cmap"].set_xlabel(LABEL)
        axs["cmap"].set_ylabel("$N$ turns")
        # set the ticks on the y axis to be the times
        plt.sca(axs["cmap"])
        plt.yticks(np.arange(len(picked_times))+0.5, labels)

        axs["hist1"].hist(np.log10(DATA[picked_times[-1]].to_numpy()), bins=50, range=(val_min, val_max), density=True)
        axs["hist2"].hist(np.log10(DATA[picked_times[len(picked_times)//2]].to_numpy()), bins=50, range=(val_min, val_max), density=True)
        axs["hist3"].hist(np.log10(DATA[picked_times[0]].to_numpy()), bins=50, range=(val_min, val_max), density=True)

        axs["hist1"].axvline(
            TH_BEST[corresponding_indexes[-1]], c="aqua", linestyle="--", label="A posteriori threshold")
        axs["hist1"].axvline(
            TH_CHOSEN[corresponding_indexes[-1]], c="r", linestyle="--", label="Basic KMeans threshold")
        axs["hist1"].axvline(
            TH_F1[corresponding_indexes[-1]], c="grey", linestyle="--", label="Best F1 threshold")

        axs["hist2"].axvline(
            TH_BEST[corresponding_indexes[len(picked_times)//2]], c="aqua", linestyle="--", label="A posteriori threshold")
        axs["hist2"].axvline(
            TH_CHOSEN[corresponding_indexes[len(picked_times)//2]], c="r", linestyle="--", label="Basic KMeans threshold")
        axs["hist2"].axvline(
            TH_F1[corresponding_indexes[len(picked_times)//2]], c="grey", linestyle="--", label="Best F1 threshold")

        axs["hist3"].axvline(
            TH_BEST[corresponding_indexes[0]], c="aqua", linestyle="--", label="A posteriori threshold")
        axs["hist3"].axvline(
            TH_CHOSEN[corresponding_indexes[0]], c="r", linestyle="--", label="Basic KMeans threshold")
        axs["hist3"].axvline(
            TH_F1[corresponding_indexes[0]], c="grey", linestyle="--", label="Best F1 threshold")

        axs["hist1"].legend(fontsize="small")
        axs["hist2"].legend(fontsize="small")
        axs["hist3"].legend(fontsize="small")

        axs["hist1"].set_ylabel("Frequency")
        axs["hist2"].set_ylabel("Frequency")
        axs["hist3"].set_ylabel("Frequency")
        axs["hist3"].set_xlabel(LABEL)
        # move ylabels to the right
        axs["hist1"].yaxis.tick_right()
        axs["hist2"].yaxis.tick_right()
        axs["hist3"].yaxis.tick_right()
        # move y title to the right
        axs["hist1"].yaxis.set_label_position("right")
        axs["hist2"].yaxis.set_label_position("right")
        axs["hist3"].yaxis.set_label_position("right")

        # set titles for the histos
        axs["hist1"].set_title(f"{labels[-1]} turns")
        axs["hist2"].set_title(f"{labels[len(picked_times)//2]} turns")
        axs["hist3"].set_title(f"{labels[0]} turns")

        for pos, idx in enumerate(corresponding_indexes):
            axs["cmap"].plot([TH_BEST[idx], TH_BEST[idx]], [pos, pos+1], c="aqua", linestyle="--")
            axs["cmap"].plot([TH_CHOSEN[idx], TH_CHOSEN[idx]], [pos, pos+1], c="r", linestyle="--")
            axs["cmap"].plot([TH_F1[idx], TH_F1[idx]], [pos, pos+1], c="grey", linestyle="--")

        d_rect_x = (val_max - val_min) * 0.005
        d_rect_y = 0.01
        rect = patches.Rectangle((val_min+d_rect_x, len(picked_times)-1+d_rect_y), val_max -
                                val_min-d_rect_x*2, 1-d_rect_y*2, linewidth=3, edgecolor='r', facecolor='none')
        axs["cmap"].add_patch(rect)

        rect = patches.Rectangle((val_min+d_rect_x, 0+d_rect_y), val_max - val_min-d_rect_x*2, 1-d_rect_y*2, linewidth=3, edgecolor='r', facecolor='none')
        axs["cmap"].add_patch(rect)


        rect = patches.Rectangle((val_min+d_rect_x, len(picked_times)//2+d_rect_y), val_max -
                                val_min-d_rect_x*2, 1-d_rect_y*2, linewidth=3, edgecolor='r', facecolor='none')
        axs["cmap"].add_patch(rect)

        # clear completely axs["cbar"]
        fig.delaxes(axs["cbar"])

        # create colorbar with height equal to axs["cmap"]
        fig.colorbar(mappable, ax=axs["cbar"],
                    location="left", label="Frequency $[\\log_{10}]$", aspect=20)
        plt.tight_layout()

        plt.savefig(os.path.join(FIGPATH, TITLE + ".jpg"), dpi=300)

    # close all figures to avoid memory leak
    plt.close("all")
