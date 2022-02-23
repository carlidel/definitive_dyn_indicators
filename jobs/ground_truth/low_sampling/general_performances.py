import numpy as np
import definitive_dyn_indicators as dd
import os

from definitive_dyn_indicators.scripts.data_manager import data_manager, refresh_henon_config

if __name__ == '__main__':
    
    group = (
        0.168,                  # omega_x
        0.201,                  # omega_y
        "sps",                  # modulation_kind
        16.0,                   # epsilon
        0.01,                   # mu
        np.nan,                 # kick amplitude
        np.nan,                 # omega_0 
    )

    for i, t_exp in enumerate([3, 4, 5, 6]):
        print(f"Executing 10^{t_exp} samples...")

        # check if the folder exists, if not, create it
        if not os.path.exists(f"10_{t_exp}"):
            os.makedirs(f"10_{t_exp}")

        print("Initializing data manager...")
        dm = data_manager(data_dir=f"./10_{t_exp}")

        print("Setting up configuration...")
        dm.henon_config["samples"] = 100
        
        dm.henon_config["t_base_2"] = np.array([], dtype=int)
        dm.henon_config["t_base"] = np.arange(10**(t_exp-3), 10**(t_exp) + 10**(t_exp-3), 10**(t_exp-3), dtype=int)
        
        dm.henon_config["t_base_10"] = np.array([], dtype=int)
        dm.henon_config["t_linear"] = np.array([], dtype=int)

        dm.henon_config = refresh_henon_config(dm.henon_config)

        print("Running the various dynamic indicators...")
        
        dm.get_file_from_group(group, "random", "true_displacement")
        
        dm.get_file_from_group(group, "x", "true_displacement")
        dm.get_file_from_group(group, "px", "true_displacement")
        dm.get_file_from_group(group, "y", "true_displacement")
        dm.get_file_from_group(group, "py", "true_displacement")

        dm.get_file_from_group(group, "none", "track_and_reverse")

