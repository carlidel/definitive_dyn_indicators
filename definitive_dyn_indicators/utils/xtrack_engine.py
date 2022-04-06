import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import os
import pathlib
import json
from tqdm import tqdm
import h5py
from typing import List, Any
import copy

from dataclasses import dataclass

import xobjects as xo
from xobjects import context_cpu
import xtrack as xt
import xpart as xp
import definitive_dyn_indicators.scripts.dynamic_indicators as di

from cpymad.madx import Madx


@njit
def birkhoff_weights(n_weights: int) -> np.ndarray:
    weights = np.empty(n_weights)
    for i in range(n_weights):
        t = i / n_weights
        weights[i] = np.exp(-1 / (t * (1.0 - t)))
    weights /= np.sum(weights)
    return weights


@njit
def birkhoff_tune(x: np.ndarray, px: np.ndarray):
    size = x.shape[1]
    x_px = x + 1j * px
    angle = np.angle(x_px)
    diff_angle = np.diff(angle)
    diff_angle[diff_angle < 0] += 2 * np.pi
    birkhoff_weights = birkhoff_weights(size)
    sum = np.sum(birkhoff_weights * diff_angle, axis=1)
    return 1 - sum / (2 * np.pi)


def get_lhc_mask(beam_type=1, seed=1):
    if beam_type == 1:
        beam_type = "b1_without_bb"
    elif beam_type in (2, 4):
        beam_type = "b4_without_bb"
    else:
        raise ValueError("beam_type must be 1, 2 or 4")
    position_of_this_file = pathlib.Path(__file__).parent.absolute()
    # go up two levels
    path = position_of_this_file.parent.parent
    # go to the masks folder
    path = path.joinpath('masks')
    # list files in the path
    files = os.listdir(path)
    filename = "lhc_mask_" + beam_type + "_" + str(seed) + ".json"
    if filename in files:
        # return complete path of the file
        return path.joinpath(filename)
    else:
        raise Exception("Mask not found!")


def sample_4d_spherical_direction(n_samples):
    """
    Sample 4D spherical directions.
    """
    # sample 4D directions
    directions = np.random.randn(n_samples, 4)
    # normalize directions
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    return directions


def sample_6d_spherical_direction(n_samples):
    """
    Sample 6D spherical directions.
    """
    # sample 6D directions
    directions = np.random.randn(n_samples, 6)
    # normalize directions
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    return directions


@dataclass
class LHCConfig:
    config_name: str
    beam_number: int
    beam_seed: int
    mask_path: str = "../../masks/lhc_mask_b1_without_bb_1.json"

    def __post_init__(self):
        self.mask_path = get_lhc_mask(self.beam_number, self.beam_seed)

    def get_tracker(self, context=xo.ContextCpu()):
        with open(self.mask_path) as f:
            line_data = json.load(f)
        sequence = xt.Line.from_dict(line_data)
        return xt.Tracker(_context=context, line=sequence)


@dataclass
class ParticlesData:
    x: np.ndarray
    px: np.ndarray
    y: np.ndarray
    py: np.ndarray
    zeta: np.ndarray
    delta: np.ndarray
    steps: np.ndarray

    def create_particles(self,  p0c=7000e9, context=xo.ContextCpu()):
        particles = xp.Particles(
            x=self.x,
            px=self.px,
            y=self.y,
            py=self.py,
            zeta=self.zeta,
            delta=self.delta,
            at_turn=self.steps,
            p0c=p0c,
            _context=context
        )
        return particles

@dataclass
class ParticlesConfig:
    samples: int

    x_min: float
    x_max: float
    
    y_min: float
    y_max: float

    zeta_value: float = 0.0 # 0.0, 0.15, 0.30        

    @property
    def total_samples(self) -> int:
        return self.samples ** 2

    def get_initial_conditions(self):
        xx, yy = np.meshgrid(
            np.linspace(self.x_min, self.x_max, self.samples),
            np.linspace(self.y_min, self.y_max, self.samples),
        )
        x_flat = xx.flatten()
        y_flat = yy.flatten()
        px_flat = np.zeros_like(x_flat)
        py_flat = np.zeros_like(x_flat)
        
        z_flat = np.ones_like(x_flat) * self.zeta_value
        delta_flat = np.zeros_like(x_flat)

        return ParticlesData(
            x=x_flat, px=px_flat, y=y_flat, py=py_flat,
            zeta=z_flat, delta=delta_flat, steps=np.zeros_like(x_flat))

    def get_initial_conditions_with_displacement(self, displacement_module: float, displacement_kind: str):
        p_data = self.get_initial_conditions()
        
        if displacement_kind == "x":
            p_data.x += displacement_module
        elif displacement_kind == "px":
            p_data.px += displacement_module
        elif displacement_kind == "y":
            p_data.y += displacement_module
        elif displacement_kind == "py":
            p_data.py += displacement_module
        elif displacement_kind == "z":
            p_data.zeta += displacement_module
        elif displacement_kind == "delta":
            p_data.delta += displacement_module
        elif displacement_kind == "random_4d":
            directions = sample_4d_spherical_direction(p_data.x.size)
            p_data.x += displacement_module * directions[:, 0]
            p_data.px += displacement_module * directions[:, 1]
            p_data.y += displacement_module * directions[:, 2]
            p_data.py += displacement_module * directions[:, 3]
        elif displacement_kind == "random":
            directions = sample_6d_spherical_direction(p_data.x.size)
            p_data.x += displacement_module * directions[:, 0]
            p_data.px += displacement_module * directions[:, 1]
            p_data.y += displacement_module * directions[:, 2]
            p_data.py += displacement_module * directions[:, 3]
            p_data.zeta += displacement_module * directions[:, 4]
            p_data.delta += displacement_module * directions[:, 5]
        else:
            raise ValueError("displacement_kind must be x, px, y, py, z, delta or random")
        return p_data


@dataclass
class RunConfig:
    times: np.ndarray
    t_norm: int
    displacement_module: float

    t_checkpoints: int = 10000

    def get_event_list(self, samples=True, normalize=True, checkpoints=True):
        self.times = np.sort(self.times)
        event_list = []
        if samples:
            event_list += [
                ["sample", t] for t in self.times
            ]
        if normalize:
            event_list += [
                ["normalize", t] for t in np.arange(self.t_norm, np.max(self.times), self.t_norm)
            ]
        if checkpoints:
            event_list += [
                ["checkpoint", t] for t in np.arange(self.t_checkpoints, np.max(self.times), self.t_checkpoints)
            ]
        # sort by t
        event_list = list(sorted(event_list, key=lambda x: x[1] + (0.0 if x[0] == "sample" else 0.1 if x[0] == "normalize" else 0.2)))
        return event_list
    
    def copy(self):
        return copy.deepcopy(self)


@dataclass
class Checkpoint:
    particles_config: ParticlesConfig
    lhc_config: LHCConfig
    run_config: RunConfig
    
    particles_list: List[dict]
    current_t: int = 0
    completed: bool = False

    def __repr__(self) -> str:
        return f"Checkpoint(current_t={self.current_t}, particles_config={self.particles_config}, lhc_config={self.lhc_config}, run_config={self.run_config})"


def get_particle_data(particles: xp.Particles, context=xo.ContextCpu()):
    x = context.nparray_from_context_array(particles.x)
    px = context.nparray_from_context_array(particles.px)
    y = context.nparray_from_context_array(particles.y)
    py = context.nparray_from_context_array(particles.py)
    zeta = context.nparray_from_context_array(particles.zeta)
    delta = context.nparray_from_context_array(particles.delta)
    at_turn = context.nparray_from_context_array(particles.at_turn)
    particle_id = context.nparray_from_context_array(particles.particle_id)

    argsort = particle_id.argsort()
    n_turns = int(np.max(at_turn))

    x = x[argsort]
    px = px[argsort]
    y = y[argsort]
    py = py[argsort]
    zeta = zeta[argsort]
    delta = delta[argsort]
    at_turn = at_turn[argsort]

    x[at_turn < n_turns] = np.nan
    px[at_turn < n_turns] = np.nan
    y[at_turn < n_turns] = np.nan
    py[at_turn < n_turns] = np.nan
    zeta[at_turn < n_turns] = np.nan
    delta[at_turn < n_turns] = np.nan

    return ParticlesData(x=x, px=px, y=y, py=py, zeta=zeta, delta=delta, steps=at_turn)


def realign_particles(particles_ref: xp.Particles, particles_target: xp.Particles, module: float, realign_4d_only: bool, context=xo.ContextCpu()):
    p_ref = get_particle_data(particles_ref, context=context)
    p_target = get_particle_data(particles_target, context=context)

    if realign_4d_only:
       distance = np.sqrt((p_ref.x - p_target.x)**2 + (p_ref.px - p_target.px)**2 + (p_ref.y - p_target.y)**2 + (p_ref.py - p_target.py)**2)
    else:
        distance = np.sqrt((p_ref.x - p_target.x)**2 + (p_ref.px - p_target.px)**2 + (p_ref.y - p_target.y)**2 + (p_ref.py - p_target.py)**2 + (p_ref.zeta - p_target.zeta)**2 + (p_ref.delta - p_target.delta)**2)

    ratio = module / distance

    particles_target.x = context.nparray_to_context_array(
        p_ref.x + (p_target.x - p_ref.x) * ratio)
    particles_target.px = context.nparray_to_context_array(
        p_ref.px + (p_target.px - p_ref.px) * ratio)
    particles_target.y = context.nparray_to_context_array(
        p_ref.y + (p_target.y - p_ref.y) * ratio)
    particles_target.py = context.nparray_to_context_array(
        p_ref.py + (p_target.py - p_ref.py) * ratio)
    if not realign_4d_only:
        particles_target.zeta = context.nparray_to_context_array(
            p_ref.zeta + (p_target.zeta - p_ref.zeta) * ratio)
        particles_target.delta = context.nparray_to_context_array(
            p_ref.delta + (p_target.delta - p_ref.delta) * ratio)


def get_displacement_module(particles_1: xp.Particles, particles_2: xp.Particles, context=xo.ContextCpu()):
    p_1 = get_particle_data(particles_1, context=context)
    p_2 = get_particle_data(particles_2, context=context)

    distance = np.sqrt((p_1.x - p_2.x)**2 + (p_1.px - p_2.px)**2 + (p_1.y - p_2.y)**2 + (p_1.py - p_2.py)**2 + (p_1.zeta - p_2.zeta)**2 + (p_1.delta - p_2.delta)**2)
    
    return distance


def get_displacement_direction(particles_1: xp.Particles, particles_2: xp.Particles, context=xo.ContextCpu()):
    p_1 = get_particle_data(particles_1, context=context)
    p_2 = get_particle_data(particles_2, context=context)

    direction = np.array([p_1.x - p_2.x, p_1.px - p_2.px, p_1.y - p_2.y, p_1.py - p_2.py, p_1.zeta - p_2.zeta, p_1.delta - p_2.delta])
    direction /= np.linalg.norm(direction, axis=0)

    return direction


def track_lyapunov(chk: Checkpoint, hdf5_path: str, context=xo.ContextCpu()):
    assert(len(chk.particles_list) == 2)
    if chk.current_t == 0:
        chk.displacement = np.zeros(chk.particles_config.total_samples)
    tracker = chk.lhc_config.get_tracker(context)
    p = xp.Particles.from_dict(chk.particles_list[0], _context=context)
    p_disp = xp.Particles.from_dict(chk.particles_list[1], _context=context)

    loop_start = chk.current_t
    for kind, time in tqdm(chk.run_config.get_event_list()):
        print(f"Event {kind}, at time {time}. Current time {chk.current_t}.")
        if loop_start == time:
            continue
        if chk.current_t != time:
            if time < chk.current_t:
                continue
            delta_t = time - chk.current_t
            tracker.track(p, num_turns=delta_t)
            tracker.track(p_disp, num_turns=delta_t)
            chk.current_t = time
    
        if kind == "normalize":
            chk.displacement += np.log(
                get_displacement_module(p, p_disp, context=context) / chk.run_config.displacement_module
            )
            realign_particles(
                p, p_disp, chk.run_config.displacement_module, realign_4d_only=False, context=context)

        elif kind == "checkpoint":
            chk.particles_list[0] = p.to_dict()
            chk.particles_list[1] = p_disp.to_dict()
            return chk

        elif kind == "sample":
            disp_to_save = chk.displacement + np.log(
                get_displacement_module(
                    p, p_disp, context=context) / chk.run_config.displacement_module
            )
            with h5py.File(hdf5_path, "a") as hdf5_file:
                if f"lyapunov/{time}" not in hdf5_file:                    
                    hdf5_file.create_dataset(
                        f"lyapunov/{time}", data=disp_to_save/time, compression="gzip", shuffle=True)

    with h5py.File(hdf5_path, "a") as hdf5_file:
        data = get_particle_data(p, context=context)
        if f"steps" not in hdf5_file:                    
            hdf5_file.create_dataset(
                f"steps", data=data.steps, compression="gzip", shuffle=True)
    chk.completed = True
    return chk


def track_ortho_lyapunov(chk: Checkpoint, hdf5_path: str, context=xo.ContextCpu()):
    tracker = chk.lhc_config.get_tracker(context)
    p_list = [xp.Particles.from_dict(p_data, _context=context) for p_data in chk.particles_list]

    if chk.current_t == 0:
        chk.displacement = np.zeros((len(p_list)-1, chk.particles_config.total_samples))
    disp_to_save = np.zeros((len(p_list)-1, chk.particles_config.total_samples))

    loop_start = chk.current_t
    for kind, time in tqdm(chk.run_config.get_event_list()):
        if loop_start == time:
            continue
        if chk.current_t != time:
            if time < chk.current_t:
                continue
            delta_t = time - chk.current_t
            for p in p_list:
                tracker.track(p, num_turns=delta_t)
            chk.current_t = time
        
        if kind == "normalize":
            for i, p in enumerate(p_list[1:]):
                chk.displacement[i] += np.log(
                    get_displacement_module(p, p_list[0], context=context) / chk.run_config.displacement_module
                )
            realign_particles(p_list[0], p, chk.run_config.displacement_module, realign_4d_only=False, context=context)

        elif kind == "checkpoint":
            chk.particles_list = [p.to_dict() for p in p_list]
            return chk

        elif kind == "sample":
            for i, p in enumerate(p_list[1:]):
                disp_to_save[i] = chk.displacement[i] + np.log(
                    get_displacement_module(p, p_list[0], context=context) / chk.run_config.displacement_module
                )
            with h5py.File(hdf5_path, "a") as hdf5_file:
                if f"lyapunov/{time}" not in hdf5_file:                    
                    hdf5_file.create_dataset(
                        "lyapunov/{}".format(time), data=disp_to_save/time, compression="gzip", shuffle=True)
    chk.completed = True
    return chk


def track_reverse(chk: Checkpoint, hdf5_path: str, context=xo.ContextCpu()):
    tracker = chk.lhc_config.get_tracker(context)
    backtracker = tracker.get_backtracker(_context=context)
    p = xp.Particles.from_dict(chk.particles_list[0], _context=context)

    loop_start = chk.current_t
    for kind, time in tqdm(chk.run_config.get_event_list(normalize=False)):
        if loop_start == time:
            continue
        if chk.current_t != time:
            if time < chk.current_t:
                continue
            delta_t = time - chk.current_t
            tracker.track(p, num_turns=delta_t)
            
        if kind == "checkpoint":
            chk.particles_list[0] = p.to_dict()
            return chk

        elif kind == "sample":
            p_copy = p.copy()
            backtracker.track(p_copy, num_turns=time)
            chk.current_t = time
            data_0 = chk.particles_config.get_initial_conditions()
            data_1 = get_particle_data(p_copy, context=context)
            distance = np.sqrt((data_0.x - data_1.x)**2 + (data_0.px - data_1.px)**2 + (data_0.y - data_1.y)**2 + (data_0.py - data_1.py)**2 + (data_0.zeta - data_1.zeta)**2 + (data_0.delta - data_1.delta)**2)
            
            with h5py.File(hdf5_path, "a") as hdf5_file:
                if f"reverse/{time}" not in hdf5_file:                    
                    hdf5_file.create_dataset(
                        f"reverse/{time}", data=distance, compression="gzip", shuffle=True)
    chk.completed = True
    return chk


def track_sali(chk: Checkpoint, hdf5_path: str, context=xo.ContextCpu()):
    tracker = chk.lhc_config.get_tracker(context)
    
    p = xp.Particles.from_dict(chk.particles_list[0], _context=context)
    p_x = xp.Particles.from_dict(chk.particles_list[1], _context=context)
    p_y = xp.Particles.from_dict(chk.particles_list[2], _context=context)

    d_x = get_displacement_direction(p, p_x, context=context)
    d_y = get_displacement_direction(p, p_y, context=context)
    sali = di.smallest_alignment_index_6d(
        d_x[0, :], d_x[1, :], d_x[2, :], d_x[3, :], d_x[4, :], d_x[5, :],
        d_y[0, :], d_y[1, :], d_y[2, :], d_y[3, :], d_y[4, :], d_y[5, :])

    loop_start = chk.current_t
    for kind, time in tqdm(chk.run_config.get_event_list()):
        if loop_start == time:
            continue
        if chk.current_t != time:
            if time < chk.current_t:
                continue
            delta_t = time - chk.current_t
            tracker.track(p, num_turns=delta_t)
            tracker.track(p_x, num_turns=delta_t)
            tracker.track(p_y, num_turns=delta_t)
            chk.current_t = time
        
        if kind == "normalize":
            realign_particles(p, p_x, chk.run_config.displacement_module, realign_4d_only=False, context=context)
            realign_particles(p, p_y, chk.run_config.displacement_module, realign_4d_only=False, context=context)

        if kind == "checkpoint":
            chk.particles_list[0] = p.to_dict()
            chk.particles_list[1] = p_x.to_dict()
            chk.particles_list[2] = p_y.to_dict()
            return chk

        elif kind == "sample":
            d_x = get_displacement_direction(p, p_x, context=context)
            d_y = get_displacement_direction(p, p_y, context=context)
            sali = di.smallest_alignment_index_6d(
                d_x[0, :], d_x[1, :], d_x[2, :], d_x[3, :], d_x[4, :], d_x[5, :],
                d_y[0, :], d_y[1, :], d_y[2, :], d_y[3, :], d_y[4, :], d_y[5, :])
            with h5py.File(hdf5_path, "a") as hdf5_file:
                if f"sali/{time}" not in hdf5_file:
                    hdf5_file.create_dataset(
                        f"sali/{time}", data=sali, compression="gzip", shuffle=True)
    chk.completed = True
    return chk


def track_gali_4(chk: Checkpoint, hdf5_path: str, context=xo.ContextCpu()):
    assert(len(chk.particles_list) == 5)
    tracker = chk.lhc_config.get_tracker(context)
    p_list = [xp.Particles.from_dict(p_data, _context=context)
              for p_data in chk.particles_list]
    
    d_x = get_displacement_direction(p_list[0], p_list[1], context=context)
    d_px = get_displacement_direction(p_list[0], p_list[2], context=context)
    d_y = get_displacement_direction(p_list[0], p_list[3], context=context)
    d_py = get_displacement_direction(p_list[0], p_list[4], context=context)
    gali = di.global_alignment_index_4_6d(
        d_x[0, :], d_x[1, :], d_x[2, :], d_x[3, :], d_x[4, :], d_x[5, :],
        d_px[0, :], d_px[1, :], d_px[2, :], d_px[3, :], d_px[4, :], d_px[5, :],
        d_y[0, :], d_y[1, :], d_y[2, :], d_y[3, :], d_y[4, :], d_y[5, :],
        d_py[0, :], d_py[1, :], d_py[2, :], d_py[3, :], d_py[4, :], d_py[5, :])
    
    loop_start = chk.current_t
    for kind, time in tqdm(chk.run_config.get_event_list()):
        if loop_start == time:
            continue
        if chk.current_t != time:
            if time < chk.current_t:
                continue
            delta_t = time - chk.current_t
            for p in p_list:
                tracker.track(p, num_turns=delta_t)
            chk.current_t = time

        if kind == "normalize":
            for p in p_list[1:]:
                realign_particles(p_list[0], p, chk.run_config.displacement_module, realign_4d_only=False, context=context)

        elif kind == "checkpoint":
            chk.particles_list = [p.to_dict() for p in p_list]
            return chk

        elif kind == "sample":
            d_x = get_displacement_direction(p_list[0], p_list[1], context=context)
            d_px = get_displacement_direction(p_list[0], p_list[2], context=context)
            d_y = get_displacement_direction(p_list[0], p_list[3], context=context)
            d_py = get_displacement_direction(p_list[0], p_list[4], context=context)
            gali = di.global_alignment_index_4_6d(
                d_x[0, :], d_x[1, :], d_x[2, :], d_x[3, :], d_x[4, :], d_x[5, :],
                d_px[0, :], d_px[1, :], d_px[2, :], d_px[3, :], d_px[4, :], d_px[5, :],
                d_y[0, :], d_y[1, :], d_y[2, :], d_y[3, :], d_y[4, :], d_y[5, :],
                d_py[0, :], d_py[1, :], d_py[2, :], d_py[3, :], d_py[4, :], d_py[5, :])
            with h5py.File(hdf5_path, "a") as hdf5_file:
                if f"gali/{time}" not in hdf5_file:                    
                    hdf5_file.create_dataset(
                        f"gali/{time}", data=gali, compression="gzip", shuffle=True)
    chk.completed = True
    return chk


def track_gali_6(chk: Checkpoint, hdf5_path: str, context=xo.ContextCpu()):
    assert(len(chk.particles_list) == 7)
    tracker = chk.lhc_config.get_tracker(context)
    p_list = [xp.Particles.from_dict(p_data, _context=context)
              for p_data in chk.particles_list]
    d_x = get_displacement_direction(p_list[0], p_list[1], context=context)
    d_px = get_displacement_direction(p_list[0], p_list[2], context=context)
    d_y = get_displacement_direction(p_list[0], p_list[3], context=context)
    d_py = get_displacement_direction(p_list[0], p_list[4], context=context)
    d_z = get_displacement_direction(p_list[0], p_list[5], context=context)
    d_d = get_displacement_direction(p_list[0], p_list[6], context=context)
    gali = di.global_alignment_index_6d(
        d_x[0, :], d_x[1, :], d_x[2, :], d_x[3, :], d_x[4, :], d_x[5, :],
        d_px[0, :], d_px[1, :], d_px[2, :], d_px[3, :], d_px[4, :], d_px[5, :],
        d_y[0, :], d_y[1, :], d_y[2, :], d_y[3, :], d_y[4, :], d_y[5, :],
        d_py[0, :], d_py[1, :], d_py[2, :], d_py[3, :], d_py[4, :], d_py[5, :],
        d_z[0, :], d_z[1, :], d_z[2, :], d_z[3, :], d_z[4, :], d_z[5, :],
        d_d[0, :], d_d[1, :], d_d[2, :], d_d[3, :], d_d[4, :], d_d[5, :])
    
    loop_start = chk.current_t
    for kind, time in tqdm(chk.run_config.get_event_list()):
        if loop_start == time:
            continue
        if chk.current_t != time:
            if time < chk.current_t:
                continue
            delta_t = time - chk.current_t
            for p in p_list:
                tracker.track(p, num_turns=delta_t)
            chk.current_t = time

        if kind == "normalize":
            for p in p_list[1:]:
                realign_particles(
                    p_list[0], p, chk.run_config.displacement_module, realign_4d_only=False, context=context)
        
        elif kind == "checkpoint":
            chk.particles_list = [p.to_dict() for p in p_list]
            return chk

        elif kind == "sample":
            d_x = get_displacement_direction(p_list[0], p_list[1], context=context)
            d_px = get_displacement_direction(p_list[0], p_list[2], context=context)
            d_y = get_displacement_direction(p_list[0], p_list[3], context=context)
            d_py = get_displacement_direction(p_list[0], p_list[4], context=context)
            d_z = get_displacement_direction(p_list[0], p_list[5], context=context)
            d_d = get_displacement_direction(p_list[0], p_list[6], context=context)
            gali = di.global_alignment_index_6d(
                d_x[0, :], d_x[1, :], d_x[2, :], d_x[3, :], d_x[4, :], d_x[5, :],
                d_px[0, :], d_px[1, :], d_px[2, :], d_px[3, :], d_px[4, :], d_px[5, :],
                d_y[0, :], d_y[1, :], d_y[2, :], d_y[3, :], d_y[4, :], d_y[5, :],
                d_py[0, :], d_py[1, :], d_py[2, :], d_py[3, :], d_py[4, :], d_py[5, :],
                d_z[0, :], d_z[1, :], d_z[2, :], d_z[3, :], d_z[4, :], d_z[5, :],
                d_d[0, :], d_d[1, :], d_d[2, :], d_d[3, :], d_d[4, :], d_d[5, :])
            with h5py.File(hdf5_path, "a") as hdf5_file:
                if f"gali/{time}" not in hdf5_file:
                    hdf5_file.create_dataset(
                        f"gali/{time}", data=gali, compression="gzip", shuffle=True)
    chk.completed = True
    return chk


def track_tune(p_list: List[xp.Particles], tracker: xt.Tracker, particles_config: ParticlesConfig, run_config: RunConfig, hdf5_path: str, context=xo.ContextCpu()):
    t_max = np.max(run_config.times)
    for batch, p in tqdm(enumerate(p_list), total=len(p_list)):
        tracker.track(p, num_turns=t_max, turn_by_turn_monitor=True)
        
        x = context.nparray_from_context_array(tracker.record_last_track.x)
        px = context.nparray_from_context_array(tracker.record_last_track.px)
        y = context.nparray_from_context_array(tracker.record_last_track.y)
        py = context.nparray_from_context_array(tracker.record_last_track.py)

        for i, t in enumerate(run_config.times):
            tunes_x = birkhoff_tune(x, px)
            tunes_y = birkhoff_tune(y, py)

        with h5py.File(hdf5_path, "a") as hdf5_file:
            if f"tunes_x/{batch}" not in hdf5_file:                    
                hdf5_file.create_dataset(
                    f"tunes_x/{batch}", data=tunes_x, compression="gzip", shuffle=True)
            if f"tunes_y/{batch}" not in hdf5_file:
                hdf5_file.create_dataset(
                    f"tunes_y/{batch}", data=tunes_y, compression="gzip", shuffle=True)
