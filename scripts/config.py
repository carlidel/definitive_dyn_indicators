import numpy as np

henon_square_configuration = {
    'name': 'henon_square',
    'samples': 500,
    
    'x_extents' : [0.0, 1.0],
    'y_extents' : [0.0, 1.0],
    'epsilon_list' : [0.0, 1.0, 2.0, 8.0, 16.0, 32.0, 64.0],

    'omega_x' : 0.168,
    'omega_y' : 0.201,

    'long_tracking' : 10000000,

    'frequency_tracking' : {
        'max_power_of_two' : 14,
        'min_power_of_two' : 5,
    }
}

henon_square_configuration["x_sample"] = np.linspace(
    henon_square_configuration["x_extents"][0],
    henon_square_configuration["x_extents"][1],
    henon_square_configuration["samples"]
)

henon_square_configuration["y_sample"] = np.linspace(
    henon_square_configuration["y_extents"][0],
    henon_square_configuration["y_extents"][1],
    henon_square_configuration["samples"]
)

xx, yy = np.meshgrid(
    henon_square_configuration["x_sample"],
    henon_square_configuration["y_sample"]
)

henon_square_configuration["x_flat"] = xx.flatten()
henon_square_configuration["y_flat"] = yy.flatten()
henon_square_configuration["px_flat"] = np.zeros_like(xx)
henon_square_configuration["py_flat"] = np.zeros_like(xx)

henon_square_configuration["total_samples"] = henon_square_configuration["x_flat"].size