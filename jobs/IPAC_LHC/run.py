import argparse
import os
import pickle

import definitive_dyn_indicators.utils.xtrack_engine as xe
import numpy as np
import xobjects as xo
import xpart as xp

import config as cfg

if __name__ == "__main__":

    # create parser
    parser = argparse.ArgumentParser(description="Run IPAC LHC")
    parser.add_argument("--hdf5_filename", type=str, required=True)
    parser.add_argument("--checkpoint_filename", type=str, required=True)
    parser.add_argument(
        "--hl_lhc", type=int, default=0, help="configuration index, 0-5"
    )
    parser.add_argument("--particle_config", type=int, default=0, choices=[0, 1, 2])

    parser.add_argument(
        "--dyn_ind",
        type=str,
        choices=[
            "ground_truth",
            "fli",
            "rem",
            "ofli",
            "sali",
            "gali4",
            "gali6",
            "galiraw",
            "tune",
        ],
    )
    parser.add_argument("--context", type=str, choices=["cpu", "gpu"], default="gpu")

    # parse arguments
    args = parser.parse_args()

    if args.context == "gpu":
        context = xo.ContextCupy()
    else:
        context = xo.ContextCpu()

    particle_config = cfg.particle_config_low[args.particle_config]

    eos_config = cfg.default_eos
    eos_config.hdf5_filename = args.hdf5_filename
    eos_config.checkpoint_filename = args.checkpoint_filename

    hdf5_exists, checkpoint_exists = eos_config.grab_files_from_eos(
        avoid_hdf5=True if args.dyn_ind == "tune" else False
    )

    if checkpoint_exists:
        with open(
            os.path.join(eos_config.local_path, eos_config.checkpoint_filename), "rb"
        ) as f:
            chk = pickle.load(f)
    else:
        p_list = []
        p_list.append(
            particle_config.get_initial_conditions()
            .create_particles(context=context)
            .to_dict()
        )
        if args.dyn_ind == "ground_truth" or args.dyn_ind == "fli":
            p_list.append(
                particle_config.get_initial_conditions_with_displacement(
                    cfg.run_config_dyn_indicator.displacement_module, "random_4d"
                )
                .create_particles(context=context)
                .to_dict()
            )
        elif args.dyn_ind == "rem":
            p_list.append(
                particle_config.get_initial_conditions()
                .create_particles(context=context)
                .to_dict()
            )
        elif (
            args.dyn_ind == "ofli"
            or args.dyn_ind == "gali6"
            or args.dyn_ind == "galiraw"
        ):
            p_list.append(
                particle_config.get_initial_conditions_with_displacement(
                    cfg.run_config_dyn_indicator.displacement_module, "x"
                )
                .create_particles(context=context)
                .to_dict()
            )
            p_list.append(
                particle_config.get_initial_conditions_with_displacement(
                    cfg.run_config_dyn_indicator.displacement_module, "px"
                )
                .create_particles(context=context)
                .to_dict()
            )
            p_list.append(
                particle_config.get_initial_conditions_with_displacement(
                    cfg.run_config_dyn_indicator.displacement_module, "y"
                )
                .create_particles(context=context)
                .to_dict()
            )
            p_list.append(
                particle_config.get_initial_conditions_with_displacement(
                    cfg.run_config_dyn_indicator.displacement_module, "py"
                )
                .create_particles(context=context)
                .to_dict()
            )
            p_list.append(
                particle_config.get_initial_conditions_with_displacement(
                    cfg.run_config_dyn_indicator.displacement_module, "z"
                )
                .create_particles(context=context)
                .to_dict()
            )
            p_list.append(
                particle_config.get_initial_conditions_with_displacement(
                    cfg.run_config_dyn_indicator.displacement_module, "delta"
                )
                .create_particles(context=context)
                .to_dict()
            )
        elif args.dyn_ind == "gali4":
            p_list.append(
                particle_config.get_initial_conditions_with_displacement(
                    cfg.run_config_dyn_indicator.displacement_module, "x"
                )
                .create_particles(context=context)
                .to_dict()
            )
            p_list.append(
                particle_config.get_initial_conditions_with_displacement(
                    cfg.run_config_dyn_indicator.displacement_module, "px"
                )
                .create_particles(context=context)
                .to_dict()
            )
            p_list.append(
                particle_config.get_initial_conditions_with_displacement(
                    cfg.run_config_dyn_indicator.displacement_module, "y"
                )
                .create_particles(context=context)
                .to_dict()
            )
            p_list.append(
                particle_config.get_initial_conditions_with_displacement(
                    cfg.run_config_dyn_indicator.displacement_module, "py"
                )
                .create_particles(context=context)
                .to_dict()
            )
        elif args.dyn_ind == "sali":
            p_list.append(
                particle_config.get_initial_conditions_with_displacement(
                    cfg.run_config_dyn_indicator.displacement_module, "x"
                )
                .create_particles(context=context)
                .to_dict()
            )
            p_list.append(
                particle_config.get_initial_conditions_with_displacement(
                    cfg.run_config_dyn_indicator.displacement_module, "y"
                )
                .create_particles(context=context)
                .to_dict()
            )

        chk = xe.Checkpoint(
            particles_config=particle_config,
            lhc_config=cfg.lhc_configs[args.hl_lhc],
            run_config=cfg.run_config_ground_truth
            if args.dyn_ind == "ground_truth"
            else cfg.run_config_reverse_indicator
            if args.dyn_ind == "rem"
            else cfg.run_config_tune
            if args.dyn_ind == "tune"
            else cfg.run_config_dyn_indicator,
            particles_list=p_list,
        )

    if chk.completed:
        print("Checkpoint already completed!")
        print(eos_config)
        # Let's make it more rough with the exit
        # so that the pipeline breaks on purpose
        raise ValueError("Checkpoint already completed!")

    if args.dyn_ind == "ground_truth" or args.dyn_ind == "fli":
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

    elif args.dyn_ind == "galiraw":
        chk = xe.track_galiraw(chk, eos_config.hdf5_path(), context)

    elif args.dyn_ind == "tune":
        chk = xe.track_tune(chk, eos_config.hdf5_path(), context)

    with open(
        os.path.join(eos_config.local_path, eos_config.checkpoint_filename), "wb"
    ) as f:
        pickle.dump(chk, f)

    print("pushing back to eos...")
    print(eos_config)

    eos_config.push_files_to_eos(
        all_hdf5_files=True if args.dyn_ind == "tune" else False
    )

