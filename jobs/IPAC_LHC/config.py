import numpy as np
import os
from dataclasses import dataclass
import definitive_dyn_indicators.utils.xtrack_engine as xe

@dataclass
class EOSConfig:
    eos_path: str
    hdf5_filename: str
    checkpoint_filename: str
    local_path: str = "."

    def grab_files_from_eos(self):
        if os.path.exists(os.path.join(self.eos_path, self.hdf5_filename)):
            os.system(
                f"xrdcp root://eosuser.cern.ch/{self.eos_path}/{self.checkpoint_filename} {self.local_path}")
            eos_exists = True
        else:
            eos_exists = False

        if os.path.exists(os.path.join(self.eos_path, self.hdf5_filename)):
            os.system(
                f"xrdcp root://eosuser.cern.ch/{self.eos_path}/{self.hdf5_filename} {self.local_path}")
            checkpoint_exists = True
        else:
            checkpoint_exists = False
        
        return eos_exists, checkpoint_exists

    def push_files_to_eos(self):
        os.system(
            f"xrdcp {self.local_path}/{self.hdf5_filename} root://eosuser.cern.ch/{self.eos_path}/{self.hdf5_filename}")
        os.system(
            f"xrdcp {self.local_path}/{self.checkpoint_filename} root://eosuser.cern.ch/{self.eos_path}/{self.checkpoint_filename}")

    def hdf5_path(self) -> str:
        return os.path.join(self.local_path, self.hdf5_filename)
    
    def checkpoint_path(self) -> str:
        return os.path.join(self.local_path, self.checkpoint_filename)


default_eos = EOSConfig(
    eos_path="/eos/project/d/da-and-diffusion-studies/DA_Studies/Simulations/Models/dynamic_indicator_analysis/IPAC_LHC",
    hdf5_filename="LHC_basic.hdf5",
    checkpoint_filename="LHC_basic.pkl",
)


lhc_configs = [
    xe.LHCConfig("Beam 1, worse stability", 1, 33),
    xe.LHCConfig("Beam 1, average stability", 1, 11),
    xe.LHCConfig("Beam 1, best stability", 1, 21),
    xe.LHCConfig("Beam 2, worse stability", 2, 55),
    xe.LHCConfig("Beam 2, average stability", 2, 18),
    xe.LHCConfig("Beam 2, best stability", 2, 38),
]

particle_config_low = [
    xe.ParticlesConfig(
        samples=100,
        x_min=0, x_max=2e-3, y_min=0, y_max=2e-3,
        zeta_value=z
    ) for z in [0.0, 0.15, 0.30]
]

particle_config_mid = [
    xe.ParticlesConfig(
        samples=250,
        x_min=0, x_max=2e-3, y_min=0, y_max=2e-3,
        zeta_value=z
    ) for z in [0.0, 0.15, 0.30]
]

particle_config_high = [
    xe.ParticlesConfig(
        samples=1000,
        x_min=0, x_max=2e-3, y_min=0, y_max=2e-3,
        zeta_value=z
    ) for z in [0.0, 0.15, 0.30]
]

run_config_quickest = xe.RunConfig(
    times=np.array([10, 20]),
    t_norm=10,
    t_checkpoints=10,
    displacement_module=1e-12
)

run_config_quickest_no_chk = xe.RunConfig(
    times=np.array([10, 20]),
    t_norm=10,
    t_checkpoints=1000000,
    displacement_module=1e-12
)

run_config_test = xe.RunConfig(
    times=np.arange(100, 1100, 100),
    t_norm=100,
    t_checkpoints=500,
    displacement_module=1e-12
)

run_config_test_no_chk = xe.RunConfig(
    times=np.arange(100, 1100, 100),
    t_norm=100,
    t_checkpoints=1000000,
    displacement_module=1e-12
)

run_config_dyn_indicator = xe.RunConfig(
    times=np.arange(100, 100100, 100),
    t_norm=100,
    t_checkpoints=35000,
    displacement_module=1e-12
)

run_config_ground_truth = xe.RunConfig(
    times=np.array([100, 1000, 10000, 100000, 1000000, 10000000]),
    t_norm=100,
    t_checkpoints=80000,
    displacement_module=1e-12
)
