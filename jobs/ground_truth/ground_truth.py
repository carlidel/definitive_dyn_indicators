import numpy as np
import definitive_dyn_indicators as dd

from definitive_dyn_indicators.scripts.data_manager import data_manager

if __name__ == '__main__':
    
    print("Initializing data manager...")
    dm = data_manager(data_dir = ".")

    print("Setting up configuration...")
    dm.henon_config["samples"] = 1000
    
    dm.henon_config["t_base_2"] = np.array([], dtype=int)
    dm.henon_config["t_base"] = np.array([], dtype=int)
    
    dm.henon_config["t_base_10"] = np.logspace(3, 8, 16, base=10, dtype=int)
    dm.henon_config["t_linear"] = np.linspace(100000, 100000000, 2000, dtype=int)

    dm.henon_config = dd.refresh_henon_config(dm.henon_config)

    group = (
        0.168,                  # omega_x
        0.201,                  # omega_y
        "sps",                  # modulation_kind
        16.0,                   # epsilon
        0.01,                   # mu
        np.nan,                 # kick amplitude
        np.nan,                 # omega_0 
    )

    print("Running ground truth...")
    dm.get_file_from_group(group, "random", "true_displacement")