import numpy as np
import os
import argparse
import pickle
import xpart as xp
import xobjects as xo

import definitive_dyn_indicators.utils.xtrack_engine as xe
import config as cfg


if __name__ == "__main__":

    # create parser
    parser = argparse.ArgumentParser(description="Run IPAC LHC")
    parser.add_argument("--hdf5_filename", type=str, required=True)
    parser.add_argument("--checkpoint_filename", type=str, required=True)
    parser.add_argument("--hl_lhc", type=int, default=0, help="configuration index, 0-5")

    parser.add_argument("--dyn_ind", type=str, 
        choices=["ground_truth", "fli", "rem", "ofli", "sali", "gali4", "gali6", "tune"]
    )
    parser.add_argument("--context", type=str, choices=["cpu", "gpu"], default="gpu")

    # parse arguments
    args = parser.parse_args()

    if args.context == "gpu":
        context = xo.ContextCupy()
    else:
        context = xo.ContextCpu()

    eos_config = cfg.default_eos
    eos_config.hdf5_filename = args.hdf5_filename
    eos_config.checkpoint_filename = args.checkpoint_filename

    eos_exists, checkpoint_exists = eos_config.grab_files_from_eos()

    if checkpoint_exists:
        with open(os.path.join(eos_config.local_path, eos_config.checkpoint_filename), "rb") as f:
            chk = pickle.load(f) 
    else:
        p_list = []
        p_list.append(xp.Particles(
            cfg.particle_config_low.get_initial_codintions()).to_dict())
        if args.dyn_ind == "ground_truth" or args.dyn_ind == "fli":
            p_list.append(xp.Particles(
                cfg.particle_config_mid.get_initial_conditions_with_displacement(cfg.run_config_dyn_indicator.displacement_module, "random_4d")).to_dict())
        elif args.dyn_ind == "ofli" or args.dyn_ind == "gali6":
            p_list.append(xp.Particles(
                cfg.particle_config_mid.get_initial_conditions_with_displacement(cfg.run_config_dyn_indicator.displacement_module, "x")).to_dict())
            p_list.append(xp.Particles(
                cfg.particle_config_mid.get_initial_conditions_with_displacement(cfg.run_config_dyn_indicator.displacement_module, "px")).to_dict())
            p_list.append(xp.Particles(
                cfg.particle_config_mid.get_initial_conditions_with_displacement(cfg.run_config_dyn_indicator.displacement_module, "y")).to_dict())
            p_list.append(xp.Particles(
                cfg.particle_config_mid.get_initial_conditions_with_displacement(cfg.run_config_dyn_indicator.displacement_module, "py")).to_dict())
            p_list.append(xp.Particles(
                cfg.particle_config_mid.get_initial_conditions_with_displacement(cfg.run_config_dyn_indicator.displacement_module, "z")).to_dict())
            p_list.append(xp.Particles(
                cfg.particle_config_mid.get_initial_conditions_with_displacement(cfg.run_config_dyn_indicator.displacement_module, "delta")).to_dict())
        elif args.dyn_ind == "gali4":
            p_list.append(xp.Particles(
                cfg.particle_config_mid.get_initial_conditions_with_displacement(cfg.run_config_dyn_indicator.displacement_module, "x")).to_dict())
            p_list.append(xp.Particles(
                cfg.particle_config_mid.get_initial_conditions_with_displacement(cfg.run_config_dyn_indicator.displacement_module, "px")).to_dict())
            p_list.append(xp.Particles(
                cfg.particle_config_mid.get_initial_conditions_with_displacement(cfg.run_config_dyn_indicator.displacement_module, "y")).to_dict())
            p_list.append(xp.Particles(
                cfg.particle_config_mid.get_initial_conditions_with_displacement(cfg.run_config_dyn_indicator.displacement_module, "py")).to_dict())
        elif args.dyn_ind == "sali":
            p_list.append(xp.Particles(
                cfg.particle_config_mid.get_initial_conditions_with_displacement(cfg.run_config_dyn_indicator.displacement_module, "x")).to_dict())
            p_list.append(xp.Particles(
                cfg.particle_config_mid.get_initial_conditions_with_displacement(cfg.run_config_dyn_indicator.displacement_module, "y")).to_dict())

        chk = xe.Checkpoint(
            particles_config=cfg.particle_config_low,
            lhc_config=cfg.lhc_configs[args.hl_lhc],
            run_config=cfg.run_config_ground_truth if args.dyn_ind == "ground_truth" else cfg.run_config_dyn_indicator,
            eos_config=eos_config,
            particles_list=p_list
        )

    if chk.completed:
        print("Checkpoint already completed!")
        print(eos_config)
        exit(0)

    if args.dyn_ind == "ground_truth" or args.dy_ind == "fli":
        chk = xe.track_lyapunov(chk, eos_config.hdf5_path(), context)

    elif args.dyn_ind == "ofli":
        chk = xe.track_ortho_lyapunov(chk, eos_config.hdf5_path(), context)

    elif args.dyn_ind == "rem":
        chk = xe.track_reverse(chk, eos_config.hdf5_path(), context)
    
    elif args.dyn_ind == "sali":
        chk = xe.track_sali(chk, eos_config.hdf5_path(), context)
    
    elif args.dyn_ind == "gali4":
        chk = xe.track_gali_4(chk, eos_config.hdf5_path(), context)

    elif args.dyn_ind == "gali6":
        chk = xe.track_gali_6(chk, eos_config.hdf5_path(), context)

    elif args.dyn_ind == "tune":
        raise NotImplementedError("Tune not implemented yet")


    with open(os.path.join(eos_config.local_path, eos_config.checkpoint_filename), "wb") as f:
        pickle.dump(chk, f)

    print("pushing back to eos...")
    print(eos_config)

    eos_config.push_files_to_eos()