import numpy as np

lhc_config = {
    'name': 'lhc_square_test',
    'samples': 200,

    'x_extents': [0.0, 2e-3],
    'y_extents': [0.0, 2e-3],

    'tracking': 10000,
}

lhc_config["x_sample"], lhc_config["dx"] = np.linspace(
    lhc_config["x_extents"][0],
    lhc_config["x_extents"][1],
    lhc_config["samples"],
    retstep=True
)

lhc_config["y_sample"], lhc_config["dy"] = np.linspace(
    lhc_config["y_extents"][0],
    lhc_config["y_extents"][1],
    lhc_config["samples"],
    retstep=True
)

xx, yy = np.meshgrid(
    lhc_config["x_sample"],
    lhc_config["y_sample"]
)

lhc_config["x_flat"] = xx.flatten()
lhc_config["y_flat"] = yy.flatten()
lhc_config["px_flat"] = np.zeros_like(xx.flatten())
lhc_config["py_flat"] = np.zeros_like(xx.flatten())

lhc_config["total_samples"] = lhc_config["x_flat"].size
