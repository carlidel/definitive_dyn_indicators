import os
import subprocess
from tqdm import tqdm

DATADIR = "/afs/cern.ch/work/c/camontan/public/definitive_dyn_indicators/data/"
OUTDIR = "/eos/project/d/da-and-diffusion-studies/DA_Studies/Simulations/Models/dynamic_indicator_analysis/big_data"

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
        subprocess.run(
            [
                "cp", 
                f"{os.path.join(DATADIR, f)}", 
                f"{os.path.join(OUTDIR, f)}",
                "-v"
            ],
            check=True)
        # remove the file from the list finished
        print("Probably done...")
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
