import os
import pathlib

# get path of this script
script_path = pathlib.Path(__file__).parent.absolute()
script_path = script_path.parent.parent.absolute()

script_path = script_path.joinpath("masks")

masks = os.listdir(script_path)

masks = sorted([mask for mask in masks if mask.endswith(".json") and mask.startswith("lhc_mask")])

# Save to txt file
with open("lhc_masks.txt", "w") as f:
    for mask in masks:
        f.write(mask + "\n")