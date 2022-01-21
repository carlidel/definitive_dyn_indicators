import subprocess
import os
import pickle

with open("../config/global_config.pkl", "rb") as f:
    lhc_config = pickle.load(f)

with open("args.txt", "w") as f:
    for z in [0, 1, 2]:
        for mask in lhc_config["selected_masks_full"]:
            # execute python script
            f.write(
                " ".join([
                    "../../masks/{}".format(mask),
                    str(z),
                    "advanced_full", "\n"
                ])
            )