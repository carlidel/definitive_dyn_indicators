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
    'short_tracking' : 100,

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


lhc_square_test_configuration = {
    'name': 'lhc_square_test',
    'samples': 200,

    'x_extents' : [0.0, 2e-3],
    'y_extents' : [0.0, 2e-3],

    'long_tracking' : 10000,
    'short_tracking': 100,
}

lhc_square_test_configuration["x_sample"], lhc_square_test_configuration["dx"] = np.linspace(
    lhc_square_test_configuration["x_extents"][0],
    lhc_square_test_configuration["x_extents"][1],
    lhc_square_test_configuration["samples"],
    retstep=True
)

lhc_square_test_configuration["y_sample"], lhc_square_test_configuration["dy"] = np.linspace(
    lhc_square_test_configuration["y_extents"][0],
    lhc_square_test_configuration["y_extents"][1],
    lhc_square_test_configuration["samples"],
    retstep=True
)

xx, yy = np.meshgrid(
    lhc_square_test_configuration["x_sample"],
    lhc_square_test_configuration["y_sample"]
)

lhc_square_test_configuration["x_flat"] = xx.flatten()
lhc_square_test_configuration["y_flat"] = yy.flatten()
lhc_square_test_configuration["px_flat"] = np.zeros_like(xx.flatten())
lhc_square_test_configuration["py_flat"] = np.zeros_like(xx.flatten())

lhc_square_test_configuration["total_samples"] = lhc_square_test_configuration["x_flat"].size
