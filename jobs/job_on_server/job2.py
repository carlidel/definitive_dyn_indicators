import subprocess
import os
import pickle

with open("../config/global_config.pkl", "rb") as f:
    lhc_config = pickle.load(f)

with open("j2.sh", "w") as f:
    for z in [0, 1, 2]:
        for mask in lhc_config["selected_masks_full"][3:]:
            # execute python script
            f.write("echo \"Executing {}, {}\"\n".format(mask, z))
            f.write(
                " ".join([
                    "python3",
                    "../config/generic_run.py",
                    "--input",
                    "../../masks/{}".format(mask),
                    "-z",
                    str(z),
                    "-t",
                    "long", "\n"
                ])
            )
            f.write('echo "Done."\n')
