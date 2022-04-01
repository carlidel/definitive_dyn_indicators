import numpy as np
import argparse
import os
import time
import datetime
import pickle
from sklearn.metrics import pair_confusion_matrix
from tqdm import tqdm
import h5py
import pandas as pd
import shutil
import sys

import henon_map_cpp as hm

from definitive_dyn_indicators.scripts.dynamic_indicators import smallest_alignment_index, global_alignment_index


def henon_run(
    omega_x=0.168, omega_y=0.201, modulation_kind="sps", epsilon=1.0, mu=0.01, kick_module=np.nan, omega_0=np.nan,
    displacement_kind="none", tracking="lyapunov", barrier=100,
    x_flat=np.array([0.1,0.2]), px_flat=np.array([0.1,0.2]), y_flat=np.array([0.1,0.2]), py_flat=np.array([0.1,0.2]),
    disp_module=1e-10, t_list=np.array([100, 1100]), t_diff=np.array([100, 1000]), extreme_tracking=2000,
    outdir=".", force_CPU=False, t_normalization=1000, **kwargs):

    if tracking=="birkhoff_tunes":
        force_CPU = True

    particles = hm.particles(x_flat, px_flat, y_flat, py_flat, force_CPU=force_CPU)
    samples = len(x_flat)

    # Create engine
    tracker = hm.henon_tracker(
        np.max([np.max(t_list), extreme_tracking]), omega_x, omega_y, modulation_kind, omega_0, epsilon, offset=0, force_CPU=force_CPU)

    # create hdf5 file
    filename = f"henon_ox_{omega_x}_oy_{omega_y}_modulation_{modulation_kind}_eps_{epsilon}_mu_{mu}_kmod_{kick_module}_o0_{omega_0}_disp_{displacement_kind}_data_{tracking}.hdf5"

    # check if file exists, if yes, remove it
    if os.path.exists(filename):
        print(f"Removing {filename}")
        os.remove(filename)

    print(f"Creating {filename}")
    data = h5py.File(os.path.join(outdir, filename), "w")

    if tracking == "track":
        # start chronometer
        start = time.time()
        print("Tracking...")

        tracker.track(particles, extreme_tracking,
                      mu, barrier, kick_module, False)

        data.create_dataset("x", data=particles.get_x(),
                            compression="gzip", shuffle=True)
        data.create_dataset("px", data=particles.get_px(),
                            compression="gzip", shuffle=True)
        data.create_dataset("y", data=particles.get_y(),
                            compression="gzip", shuffle=True)
        data.create_dataset("py", data=particles.get_py(),
                            compression="gzip", shuffle=True)

        data.create_dataset("steps", data=particles.get_steps(),
                            compression="gzip", shuffle=True)

        # stop chronometer
        end = time.time()
        # print time in hh:mm:ss
        print(f"Elapsed time: {datetime.timedelta(seconds=end-start)}")
    elif tracking == "step_track":
        for i, (t, t_sum) in tqdm(enumerate(zip(t_diff, t_list)), total=len(t_diff)):
            tracker.track(particles, t, mu,
                          barrier, kick_module, False)

            data.create_dataset("x", data=particles.get_x(),
                                compression="gzip", shuffle=True)
            data.create_dataset("px", data=particles.get_px(),
                                compression="gzip", shuffle=True)
            data.create_dataset("y", data=particles.get_y(),
                                compression="gzip", shuffle=True)
            data.create_dataset("py", data=particles.get_py(),
                                compression="gzip", shuffle=True)

        data.create_dataset("steps", data=particles.get_steps(),
                            compression="gzip", shuffle=True)
    elif tracking == "track_and_reverse":
        for i, (t, t_sum) in tqdm(enumerate(zip(t_list, t_list)), total=len(t_list)):
            particles.reset()
            tracker.track(particles, t, mu, barrier,
                            kick_module, False)
            tracker.track(particles, t, mu, barrier,
                            kick_module, True)
                
            data.create_dataset(f"x/{t}", data=particles.get_x(),
                                compression="gzip", shuffle=True)
            data.create_dataset(f"px/{t}", data=particles.get_px(),
                                compression="gzip", shuffle=True)
            data.create_dataset(f"y/{t}", data=particles.get_y(),
                                compression="gzip", shuffle=True)
            data.create_dataset(f"py/{t}", data=particles.get_py(),
                                compression="gzip", shuffle=True)

    elif tracking == "megno":
        particles.add_ghost(disp_module, "random")
        raise NotImplementedError

    elif tracking == "lyapunov":
        particles.add_ghost(disp_module, "random")
        displacement = np.zeros_like(x_flat)
        
        event_list = [
            ["sample", t] for t in t_list
        ] + [
            ["normalize", t] for t in np.arange(t_normalization, np.max(t_list), t_normalization)
        ]
        event_list = list(sorted(event_list, key=lambda x: x[1]))
        
        current_t = 0
        for kind, time in tqdm(event_list):
            if current_t != time:
                delta_t = time - current_t
                tracker.track(particles, delta_t, mu,
                                barrier, kick_module, False)
                current_t = time

            if kind == "normalize":
                displacement += np.log(particles.get_displacement_module().flatten() / disp_module)
                particles.renormalize(disp_module)
                
            elif kind == "sample":
                disp_to_save = displacement + np.log(particles.get_displacement_module().flatten() / disp_module)
                data.create_dataset(
                    f"lyapunov/{time}", data=disp_to_save/time, compression="gzip", shuffle=True)
                data.create_dataset(
                    f"x/{time}", data=particles.get_x(), compression="gzip", shuffle=True)
                data.create_dataset(
                    f"px/{time}", data=particles.get_px(), compression="gzip", shuffle=True)
                data.create_dataset(
                    f"y/{time}", data=particles.get_y(), compression="gzip", shuffle=True)
                data.create_dataset(
                    f"py/{time}", data=particles.get_py(), compression="gzip", shuffle=True)

        data.create_dataset("steps", data=particles.get_steps(),
                            compression="gzip", shuffle=True)

    elif tracking == "orthogonal_lyapunov":
        particles.add_ghost(disp_module, "x")
        particles.add_ghost(disp_module, "px")
        particles.add_ghost(disp_module, "y")
        particles.add_ghost(disp_module, "py")

        displacement = np.zeros((x_flat.size, 4))

        event_list = [
            ["sample", t] for t in t_list
        ] + [
            ["normalize", t] for t in np.arange(t_normalization, np.max(t_list), t_normalization)
        ]
        event_list = list(sorted(event_list, key=lambda x: x[1]))

        current_t = 0
        for kind, time in tqdm(event_list):
            if current_t != time:
                delta_t = time - current_t
                tracker.track(particles, delta_t, mu,
                              barrier, kick_module, False)
                current_t = time

            if kind == "normalize":
                displacement += np.log(
                    particles.get_displacement_module() / disp_module)
                particles.renormalize(disp_module)
           
            elif kind == "sample":
                disp_to_save = displacement + np.log(particles.get_displacement_module() / disp_module)
                data.create_dataset(
                    f"lyapunov/{time}", data=disp_to_save/time, compression="gzip", shuffle=True)

    elif tracking == "lyapunov_birkhoff":
        particles.add_ghost(disp_module, "random")
        displacement = []

        event_list = [
            ["sample", t] for t in t_list
        ] + [
            ["normalize", t] for t in np.arange(t_normalization, np.max(t_list), t_normalization)
        ]
        event_list = list(sorted(event_list, key=lambda x: x[1]))
        current_t = 0

        for kind, time in tqdm(event_list):
            if kind == "sample":
                if current_t != time:
                    delta_t = time - current_t
                    tracker.track(particles, delta_t, mu,
                                  barrier, kick_module, False)
                displacement.append(np.log(particles.get_displacement_module(
                ).flatten() / disp_module))
                current_t = time
            elif kind == "normalize":
                if current_t != time:
                    delta_t = time - current_t
                    tracker.track(particles, delta_t, mu,
                                  barrier, kick_module, False)
                displacement.append(np.log(particles.get_displacement_module(
                ).flatten() / disp_module))
                particles.renormalize(displacement)
                current_t = time
            if kind == "sample":
                birkhoff_coeff = hm.birkhoff_weights(len(displacement))
                value = np.sum(np.array(displacement) * birkhoff_coeff[:, None], axis=0) / t_normalization
                data.create_dataset(
                    f"lyapunov/{time}", data=value, compression="gzip", shuffle=True)

        data.create_dataset("steps", data=particles.get_steps(),
                            compression="gzip", shuffle=True)

    elif tracking == "sali":
        particles.add_ghost(disp_module, "x")
        particles.add_ghost(disp_module, "y")

        dir = particles.get_displacement_direction()
        sali = smallest_alignment_index(
            dir[0,:,0], dir[1,:,0], dir[2,:,0], dir[3,:,0],
            dir[0,:,1], dir[1,:,1], dir[2,:,1], dir[3,:,1],
        )
        print("Dir shape:", dir.shape)

        event_list = [
            ["sample", t] for t in t_list
        ] + [
            ["normalize", t] for t in np.arange(t_normalization, np.max(t_list), t_normalization)
        ]
        event_list = list(sorted(event_list, key=lambda x: x[1]))
        current_t = 0

        for kind, time in tqdm(event_list):
            if kind == "sample":
                if current_t != time:
                    delta_t = time - current_t
                    tracker.track(particles, delta_t, mu,
                                  barrier, kick_module, False)
                current_t = time
            elif kind == "normalize":
                if current_t != time:
                    delta_t = time - current_t
                    tracker.track(particles, delta_t, mu,
                                  barrier, kick_module, False)
                particles.renormalize(disp_module)
                current_t = time

            # dir = particles.get_displacement_direction()
            # sali = np.min([sali, smallest_alignment_index(
            #     dir[0,:,0], dir[1,:,0], dir[2,:,0], dir[3,:,0],
            #     dir[0,:,1], dir[1,:,1], dir[2,:,1], dir[3,:,1],
            # )], axis=0)

            if kind == "sample":
                dir = particles.get_displacement_direction()
                sali = smallest_alignment_index(
                    dir[0,:,0], dir[1,:,0], dir[2,:,0], dir[3,:,0],
                    dir[0,:,1], dir[1,:,1], dir[2,:,1], dir[3,:,1],
                )
                data.create_dataset(
                    f"sali/{time}", data=sali, compression="gzip", shuffle=True)

    elif tracking == "gali":
        particles.add_ghost(disp_module, "x")
        particles.add_ghost(disp_module, "px")
        particles.add_ghost(disp_module, "y")
        particles.add_ghost(disp_module, "py")

        dir = particles.get_displacement_direction()
        gali = global_alignment_index(
            dir[0,:,0], dir[1,:,0], dir[2,:,0], dir[3,:,0],
            dir[0,:,1], dir[1,:,1], dir[2,:,1], dir[3,:,1],
            dir[0,:,2], dir[1,:,2], dir[2,:,2], dir[3,:,2],
            dir[0,:,3], dir[1,:,3], dir[2,:,3], dir[3,:,3],
        )
        print("Dir shape:", dir.shape)

        event_list = [
            ["sample", t] for t in t_list
        ] + [
            ["normalize", t] for t in np.arange(t_normalization, np.max(t_list), t_normalization)
        ]
        event_list = list(sorted(event_list, key=lambda x: x[1]))
        current_t = 0

        for kind, time in tqdm(event_list):
            if kind == "sample":
                if current_t != time:
                    delta_t = time - current_t
                    tracker.track(particles, delta_t, mu,
                                  barrier, kick_module, False)
                current_t = time
            elif kind == "normalize":
                if current_t != time:
                    delta_t = time - current_t
                    tracker.track(particles, delta_t, mu,
                                  barrier, kick_module, False)
                particles.renormalize(disp_module)
                current_t = time

            # dir = particles.get_displacement_direction()
            # gali = np.min([gali, global_alignment_index(
            #     dir[0,:,0], dir[1,:,0], dir[2,:,0], dir[3,:,0],
            #     dir[0,:,1], dir[1,:,1], dir[2,:,1], dir[3,:,1],
            #     dir[0,:,2], dir[1,:,2], dir[2,:,2], dir[3,:,2],
            #     dir[0,:,3], dir[1,:,3], dir[2,:,3], dir[3,:,3],
            # )], axis=0)

            if kind == "sample":
                dir = particles.get_displacement_direction()
                gali = global_alignment_index(
                    dir[0,:,0], dir[1,:,0], dir[2,:,0], dir[3,:,0],
                    dir[0,:,1], dir[1,:,1], dir[2,:,1], dir[3,:,1],
                    dir[0,:,2], dir[1,:,2], dir[2,:,2], dir[3,:,2],
                    dir[0,:,3], dir[1,:,3], dir[2,:,3], dir[3,:,3],
                )
                data.create_dataset(
                    f"gali/{time}", data=gali, compression="gzip", shuffle=True)

    elif tracking == "tangent_map":
        raise NotImplementedError

    elif tracking == "birkhoff_tunes":
        t_max = np.max(t_list)
        # find closest higher power of 2 to t_max
        power_2 = int(np.ceil(np.log2(t_max)))
        times = np.power(2, np.arange(3, power_2, 1))
        from_idx = np.concatenate([
            [0 for _ in times[:-1]],
            [t for t in times[:-1]]
        ])
        to_idx = np.concatenate([
            [t for t in times[:-1]],
            [t*2 for t in times[:-1]]
        ])
        
        tunes = tracker.tune_birkhoff(
            particles, t_max, mu, barrier, kick_module, False, from_idx, to_idx)
        
        for i in range(len(tunes)):
            data.create_dataset(f"tune_x/{tunes.iloc[i]['from']}/{tunes.iloc[i]['to']}", data=tunes.iloc[i]['tune_x'], compression="gzip", shuffle=True)
            data.create_dataset(f"tune_y/{tunes.iloc[i]['from']}/{tunes.iloc[i]['to']}", data=tunes.iloc[i]['tune_y'], compression="gzip", shuffle=True)

    elif tracking == "fft_tunes":
        t_max = np.max(t_list)
        # find closest higher power of 2 to t_max
        power_2 = int(np.ceil(np.log2(t_max)))
        times = np.power(2, np.arange(3, power_2, 1))
        from_idx = np.concatenate([
            [0 for _ in times[:-1]],
            [t for t in times[:-1]]
        ])
        to_idx = np.concatenate([
            [t for t in times[:-1]],
            [t*2 for t in times[:-1]]
        ])
        storage = hm.storage(samples * samples)
        for i in tqdm(range(np.max(t_list))):
            tracker.track(particles, 1, mu, barrier, kick_module, False)
            storage.store(particles)       
        tunes = storage.tune_fft(from_idx, to_idx, t_max)
        for i in range(len(tunes)):
            data.create_dataset(f"tune_x/{tunes.iloc[i]['from']}/{tunes.iloc[i]['to']}", data=tunes.iloc[i]['tune_x'], compression="gzip", shuffle=True)
            data.create_dataset(f"tune_y/{tunes.iloc[i]['from']}/{tunes.iloc[i]['to']}", data=tunes.iloc[i]['tune_y'], compression="gzip", shuffle=True)

    # create new dataset in file
    dataset = data.create_dataset("config", data=np.array([42, 42]))

    # fill attributes of dataset
    dataset.attrs["omega_x"] = omega_x
    dataset.attrs["omega_y"] = omega_y
    dataset.attrs["epsilon"] = epsilon
    dataset.attrs["mu"] = mu
    dataset.attrs["barrier"] = barrier
    dataset.attrs["kick_module"] = kick_module
    dataset.attrs["modulation_kind"] = modulation_kind
    dataset.attrs["omega_0"] = omega_0
    dataset.attrs["tracking"] = tracking
    dataset.attrs["displacement_kind"] = displacement_kind

    data.close()
