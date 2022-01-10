import numpy as np
import os
import itertools
import names

omega_list = [(0.168, 0.201), (0.31, 0.32)]

epsilon_list = [0.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
mu_list = [-1.0, -0.1, -0.01, -0.001, 0.0, 0.001, 0.01, 0.1, 1.0]
mu_chunks = [mu_list[i:i+2] for i in range(0, len(mu_list), 2)]

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

dagman_gpu = open("job_files/dagman_gpu.dag", "w")
dagman_cpu = open("job_files/dagman_cpu.dag", "w")

with open("henon_track_cpu.sub", "r") as sub_file:
    base_cpu = sub_file.read()

with open("henon_track.sub", "r") as sub_file:
    base_gpu = sub_file.read()

with open("move_files.sub", "r") as sub_file:
    base_move = sub_file.read()

# iterate for every combination of parameters
for q, (omega_x, omega_y) in enumerate(omega_list):
    for k, epsilon in enumerate(epsilon_list):
        for j, (mus) in enumerate(mu_chunks):
            print(mus)
            i = q * len(epsilon_list) * len(mu_chunks) + k * len(mu_chunks) + j

            f_gpu = open(os.path.join(
                script_dir, f"job_files/arguments_{i}.txt"), "w")
            f_cpu = open(os.path.join(
                script_dir, f"job_files/arguments_cpu_{i}.txt"), "w")

            name = names.get_full_name().replace(" ", "_")
            name = name.lower()
            for mu in mus:
                for tracking in ["step_track"]:
                    for displacement_kind in ["none", "x", "y", "px", "py", "random"]:
                        f_gpu.write(
                            f"{omega_x} {omega_y} {epsilon} {mu} {kick_module} {kick_sigma} {modulation_kind} {omega_0} {displacement_kind} {tracking} {name}\n"
                        )
                displacement_kind = "none"
                for tracking in ["track", "track_and_reverse"]:
                    f_gpu.write(
                        f"{omega_x} {omega_y} {epsilon} {mu} {kick_module} {kick_sigma} {modulation_kind} {omega_0} {displacement_kind} {tracking} {name}\n"
                    )
                for tracking in ["fft_tunes", "birkhoff_tunes"]:
                    f_cpu.write(
                        f"{omega_x} {omega_y} {epsilon} {mu} {kick_module} {kick_sigma} {modulation_kind} {omega_0} {displacement_kind} {tracking} {name + '_bis'}\n"
                )

            f_gpu.close()
            f_cpu.close()

            with open(f"job_files/henon_track_{i}.sub", 'w') as f:
                f.write(base_gpu.replace("NUMBERHERE", str(i)))
            
            with open(f"job_files/henon_track_cpu_{i}.sub", 'w') as f:
                f.write(base_cpu.replace("NUMBERHERE", str(i)))

            with open(f"job_files/move_files_{i}.sub", 'w') as f:
                f.write(base_move.replace("NAMEHERE", name))

            with open(f"job_files/move_files_cpu_{i}.sub", 'w') as f:
                f.write(base_move.replace("NAMEHERE", name + "_bis"))
            
            dagman_gpu.write(f"JOB JOB{i} henon_track_{i}.sub\n")
            dagman_gpu.write(f"JOB POST{i} move_files_{i}.sub\n")
            dagman_gpu.write(f"PARENT JOB{i} CHILD POST{i}\n")
            if i != 0:
                dagman_gpu.write(f"PARENT POST{i-1} CHILD JOB{i}\n")

            dagman_cpu.write(f"JOB JOB{i} henon_track_cpu_{i}.sub\n")
            dagman_cpu.write(f"JOB POST{i} move_files_cpu_{i}.sub\n")
            dagman_cpu.write(f"PARENT JOB{i} CHILD POST{i}\n")
            if i != 0:
                dagman_cpu.write(f"PARENT POST{i-1} CHILD JOB{i}\n")
