import numpy as np
import os
import itertools
import names

omega_list = [(0.168, 0.201), (0.31, 0.32)]

epsilon_list = [0.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
mu_list = [-1.0, -0.1, -0.01, -0.001, 0.0, 0.001, 0.01, 0.1, 1.0]

barrier = 10.0
kick_module = np.nan
kick_sigma = np.nan
modulation_kind = "sps"
omega_0 = np.nan
force_CPU = False

# get position of this script
script_dir = os.path.dirname(os.path.realpath(__file__))

# move to script directory
os.chdir(script_dir)

args_gpu = open("arguments_gpu.txt", "w")
args_cpu = open("arguments_cpu.txt", "w")

# iterate for every combination of parameters
for q, (omega_x, omega_y) in enumerate(omega_list):
    for k, epsilon in enumerate(epsilon_list):
        for j, mu in enumerate(mu_list):
            for tracking in ["step_track"]:
                for displacement_kind in ["none", "x", "y", "px", "py", "random"]:
                    args_gpu.write(
                        f"{omega_x} {omega_y} {epsilon} {mu} {kick_module} {kick_sigma} {modulation_kind} {omega_0} {displacement_kind} {tracking} \n"
                    )

            displacement_kind = "none"
            for tracking in ["track", "track_and_reverse", "megno"]:
                args_gpu.write(
                    f"{omega_x} {omega_y} {epsilon} {mu} {kick_module} {kick_sigma} {modulation_kind} {omega_0} {displacement_kind} {tracking} \n"
                )
            for tracking in ["fft_tunes", "birkhoff_tunes"]:
                args_cpu.write(
                    f"{omega_x} {omega_y} {epsilon} {mu} {kick_module} {kick_sigma} {modulation_kind} {omega_0} {displacement_kind} {tracking} \n"
            )