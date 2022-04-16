import argparse
import itertools
import os

import numpy as np

LHC_CONFIG_LIST = np.arange(0, 6)
ZETA_CONFIG_LIST = np.arange(0, 3)
DYN_IND_LIST = ["ground_truth", "fli", "rem", "ofli", "sali", "gali4", "gali6", "tune"]

if __name__ == "__main__":

    # check if folder "sub_folder" exists, if not, create it
    sub_folder = "sub_folder"
    if not os.path.exists(sub_folder):
        os.makedirs(sub_folder)

    launch_all_file = open(os.path.join(sub_folder, "launch_all.sh"), "w")
    launch_all_file.write("#!/bin/bash\n\n")
    launch_all_file.write("rm -v *tune*.dag.*")
    launch_all_file.write("\n\n")

    launch_all_file_GT = open(os.path.join(sub_folder, "launch_all_GT.sh"), "w")
    launch_all_file_GT.write("#!/bin/bash\n\n")
    launch_all_file_GT.write("rm -v *.dag.*")
    launch_all_file_GT.write("\n\n")

    # open and load the txt file "head_sub.txt"
    with open("head_sub.txt", "r") as f:
        head_sub = f.read()

    for LHC, ZETA, DYN in itertools.product(
        LHC_CONFIG_LIST, ZETA_CONFIG_LIST, DYN_IND_LIST
    ):
        job_name = f"LHC_{LHC}_ZETA_{ZETA}_DYN_{DYN}"

        sub_file_name = job_name + ".sub"
        hdf5_file_name = job_name + ".hdf5"
        checkpoint_file_name = job_name + ".pkl"

        dagman_file_name = job_name + ".dag"

        if DYN != "ground_truth":
            if DYN == "tune":
                launch_all_file.write("condor_submit_dag " + dagman_file_name + "\n")
        else:
            launch_all_file_GT.write("condor_submit_dag " + dagman_file_name + "\n")

        with open(os.path.join(sub_folder, sub_file_name), "w") as f:
            f.write(head_sub)
            f.write("\n\n")
            if DYN == "tune":
                f.write("RequestCpus = 2\n")
                f.write('+JobFlavour = "espresso"')
            else:
                f.write('+JobFlavour = "longlunch"')
            f.write("\n\n")

            f.write(f"hdf5_file={hdf5_file_name}\n")
            f.write(f"chk_file={checkpoint_file_name}\n")
            f.write(f"lhc_config={LHC}\n")
            f.write(f"particle_config={ZETA}\n")
            f.write(f"dyn_ind={DYN}\n")

            f.write("queue 1")

        repetitions = (
            260
            if DYN == "ground_truth"
            else 20
            if DYN == "rem"
            else 101
            if DYN == "tune"
            else 8
        )

        with open(os.path.join(sub_folder, dagman_file_name), "w") as f:
            for i in range(repetitions):
                f.write(f"JOB JOB{i} {sub_file_name}\n")
                if i != 0:
                    f.write(f"PARENT JOB{i-1} CHILD JOB{i}\n")

    launch_all_file.close()
    launch_all_file_GT.close()
