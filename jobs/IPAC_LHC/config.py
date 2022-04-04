import numpy as np
from definitive_dyn_indicators.utils.xtrack_engine import LHCConfig, ParticlesConfig, RunConfig

lhc_configs = [
    LHCConfig("Beam 1, worse stability", 1, 33),
    LHCConfig("Beam 1, average stability", 1, 11),
    LHCConfig("Beam 1, best stability", 1, 21),
    LHCConfig("Beam 2, worse stability", 2, 55),
    LHCConfig("Beam 2, average stability", 2, 18),
    LHCConfig("Beam 2, best stability", 2, 38),
]

particle_config_low = [
    ParticlesConfig(
        samples=100,
        x_min=0, x_max=2e-3, y_min=0, y_max=2e-3,
        zeta_value=z
    ) for z in [0.0, 0.15, 0.30]
]

particle_config_mid = [
    ParticlesConfig(
        samples=250,
        x_min=0, x_max=2e-3, y_min=0, y_max=2e-3,
        zeta_value=z
    ) for z in [0.0, 0.15, 0.30]
]

particle_config_high = [
    ParticlesConfig(
        samples=1000,
        x_min=0, x_max=2e-3, y_min=0, y_max=2e-3,
        zeta_value=z
    ) for z in [0.0, 0.15, 0.30]
]

run_config_quickest = RunConfig(
    times=np.array([5, 10, 15, 20]),
    t_norm=10,
    displacement_module=1e-12
)

run_config_test = RunConfig(
    times=np.arange(100, 1100, 100),
    t_norm=100,
    displacement_module=1e-12
)

run_config_dyn_indicator = RunConfig(
    times=np.arange(100, 100100, 100),
    t_norm=100,
    displacement_module=1e-12
)

run_config_ground_truth = RunConfig(
    times=np.array([100, 1000, 10000, 100000, 1000000, 10000000]),
    t_norm=100,
    displacement_module=1e-12
)
