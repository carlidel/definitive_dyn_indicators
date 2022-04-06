import argparse
import os
import numpy as np
import itertools

LHC_CONFIG_LIST = np.arange(0, 6)
ZETA_CONFIG_LIST = np.arange(0, 3)
DYN_IND_LIST = ["ground_truth", "fli", "rem",
                "ofli", "sali", "gali4", "gali6"]#, "tune"]

if __name__ == "__main__":

    # check if folder "sub_folder" exists, if not, create it
    sub_folder = "sub_folder"
    if not os.path.exists(sub_folder):
        os.makedirs(sub_folder)

    # open and load the txt file "head_sub.txt"
    with open("head_sub.txt", "r") as f:
        head_sub = f.read()

    for LHC, ZETA, DYN in itertools.product(LHC_CONFIG_LIST, ZETA_CONFIG_LIST, DYN_IND_LIST):
        job_name = f"LHC_{LHC}_ZETA_{ZETA}_DYN_{DYN}"
        
        sub_file_name = job_name + ".sub"
        hdf5_file_name = job_name + ".hdf5"
        checkpoint_file_name = job_name + ".pkl"

        dagman_file_name = job_name + ".dag"

        with open(os.path.join(sub_folder, sub_file_name), "w") as f:
            f.write(head_sub)
            f.write("\n\n")

            f.write(f"hdf5_file={hdf5_file_name}\n")
            f.write(f"chk_file={checkpoint_file_name}\n")
            f.write(f"lhc_config={LHC}\n")
            f.write(f"particle_config={ZETA}\n")
            f.write(f"dyn_ind={DYN}\n")

            f.write("queue 1")

        repetitions = 130 if DYN == "ground_truth" else 4 

        with open(os.path.join(sub_folder, dagman_file_name), "w") as f:
            for i in range(repetitions):
                f.write(f"JOB JOB{i} {sub_file_name}\n")
                if i != 0:
                    f.write(f"PARENT JOB{i-1} CHILD JOB{i}\n")
