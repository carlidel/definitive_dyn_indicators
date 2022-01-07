import os
from tqdm import tqdm

DATADIR = "/afs/cern.ch/work/c/camontan/public/definitive_dyn_indicators/data/"
OUTDIR = "/eos/project/d/da-and-diffusion-studies/DA_Studies/Simulations/Models/dynamic_indicator_analysis/big_data"
NOEOS = False

# read file finished.txt from datadir
with open(os.path.join(DATADIR, "finished.txt"), "r") as f:
    finished = f.read().splitlines()

# delete finished.txt
os.remove(os.path.join(DATADIR, "finished.txt"))

# copy the list content to a new list
files_left = finished.copy()

try:
    # iterate over the list
    for f in tqdm(finished):
        print("copying {}".format(f))
        # perform a system command
        if NOEOS:
            os.system(
                f"cp {os.path.join(DATADIR, f)} {os.path.join(OUTDIR, f)}")
        else:
            os.system(
                f"eos cp {os.path.join(DATADIR, f)} {os.path.join(OUTDIR, f)}")
        # remove the file from the list finished
        print("removing {}".format(f))
        os.remove(os.path.join(DATADIR, f))
        files_left.remove(f)
except Exception:
    print("Error:", f)
    print("Some files could not be copied.")
    # write the list to finished.txt
    with open(os.path.join(DATADIR, "finished.txt"), "a") as f:
        for f_left in files_left:
            print(f_left)
            f.write(f_left + "\n")
