import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import os
import pathlib
import json
from tqdm import tqdm
import h5py
from typing import List

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

    def get_initial_codintions(self):
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
        p_data = self.get_initial_codintions()
        
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

    def get_event_list(self):
        self.times = np.sort(self.times)
        return [
            ["sample", t] for t in self.times
        ] + [
            ["normalize", t] for t in np.arange(self.t_norm, np.max(self.times), self.t_norm)
        ]


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


def realign_particles(particles_ref: xp.Particles, particles_target: xp.Particles, module: float, context=xo.ContextCpu()):
    p_ref = get_particle_data(particles_ref, context=context)
    p_target = get_particle_data(particles_target, context=context)

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


def track_lyapunov(p: xp.Particles, p_disp: xp.Particles, tracker: xt.Tracker, particles_config: ParticlesConfig, run_config: RunConfig, hdf5_path: str, context=xo.ContextCpu()):
    current_t = 0
    displacement = np.zeros(particles_config.total_samples)
    for kind, time in tqdm(run_config.get_event_list()):
        if current_t != time:
            delta_t = time - current_t
            tracker.track(p, num_turns=delta_t)
            tracker.track(p_disp, num_turns=delta_t)
            current_t = time
    
        if kind == "normalize":
            displacement += displacement + np.log(
                get_displacement_module(p, p_disp, context=context) / run_config.displacement_module
            )
            realign_particles(p, p_disp, run_config.displacement_module, context=context)

        elif kind == "sample":
            disp_to_save = displacement + np.log(
                get_displacement_module(p, p_disp, context=context) / run_config.displacement_module
            )
            with h5py.File(hdf5_path, "a") as hdf5_file:
                hdf5_file.create_dataset(
                    f"lyapunov/{time}", data=disp_to_save/time, compression="gzip", shuffle=True)

    with h5py.File(hdf5_path, "a") as hdf5_file:
        data = get_particle_data(p, context=context)
        hdf5_file.create_dataset(
            f"steps", data=data.steps, compression="gzip", shuffle=True)


def track_ortho_lyapunov(p_list: List[xp.Particles], tracker: xt.Tracker, particles_config: ParticlesConfig, run_config: RunConfig, hdf5_path: str, context=xo.ContextCpu()):
    current_t = 0
    displacement = np.zeros((len(p_list)-1, particles_config.total_samples))
    disp_to_save = np.zeros((len(p_list)-1, particles_config.total_samples))
    for kind, time in tqdm(run_config.get_event_list()):
        if current_t != time:
            delta_t = time - current_t
            for p in p_list:
                tracker.track(p, num_turns=delta_t)
            current_t = time
        
        if kind == "normalize":
            for i, p in enumerate(p_list[1:]):
                displacement[i] += displacement[i] + np.log(
                    get_displacement_module(p, p_list[0], context=context) / run_config.displacement_module
                )
            realign_particles(p_list[0], p, run_config.displacement_module, context=context)

        elif kind == "sample":
            for i, p in enumerate(p_list[1:]):
                disp_to_save[i] += displacement[i] + np.log(
                    get_displacement_module(p, p_list[0], context=context) / run_config.displacement_module
                )
            with h5py.File(hdf5_path, "a") as hdf5_file:
                hdf5_file.create_dataset(
                    "lyapunov/{}".format(time), data=disp_to_save/time, compression="gzip", shuffle=True)


def track_reverse(p: xp.Particles, tracker: xt.Tracker, particles_config: ParticlesConfig, run_config: RunConfig, hdf5_path: str, context=xo.ContextCpu()):
    backtracker = tracker.get_backtracker(_context=context)
    current_time = 0

    data_0 = get_particle_data(p, context=context)

    for time in tqdm(run_config.times):
        delta_t = time - current_time
        tracker.track(p, num_turns=delta_t)
        p_copy = p.copy()
        backtracker.track(p_copy, num_turns=time)
        current_time = time

        data_1 = get_particle_data(p_copy, context=context)
        distance = np.sqrt((data_0.x - data_1.x)**2 + (data_0.px - data_1.px)**2 + (data_0.y - data_1.y)**2 + (data_0.py - data_1.py)**2 + (data_0.zeta - data_1.zeta)**2 + (data_0.delta - data_1.delta)**2)
        
        with h5py.File(hdf5_path, "a") as hdf5_file:
            hdf5_file.create_dataset(
                f"reverse/{time}", data=distance, compression="gzip", shuffle=True)


def track_sali(p: xp.Particles, p_x: xp.Particles, p_y: xp.Particles, tracker: xt.Tracker, particles_config: ParticlesConfig, run_config: RunConfig, hdf5_path: str, context=xo.ContextCpu()):
    current_time = 0
    d_x = get_displacement_direction(p, p_x, context=context)
    d_y = get_displacement_direction(p, p_y, context=context)
    sali = di.smallest_alignment_index_6d(
        d_x[0, :], d_x[1, :], d_x[2, :], d_x[3, :], d_x[4, :], d_x[5, :],
        d_y[0, :], d_y[1, :], d_y[2, :], d_y[3, :], d_y[4, :], d_y[5, :])

    for kind, time in tqdm(run_config.get_event_list()):
        if current_time != time:
            delta_t = time - current_time
            tracker.track(p, num_turns=delta_t)
            tracker.track(p_x, num_turns=delta_t)
            tracker.track(p_y, num_turns=delta_t)
            current_time = time
        
        if kind == "normalize":
            realign_particles(p, p_x, run_config.displacement_module, context=context)
            realign_particles(p, p_y, run_config.displacement_module, context=context)

        elif kind == "sample":
            d_x = get_displacement_direction(p, p_x, context=context)
            d_y = get_displacement_direction(p, p_y, context=context)
            sali = di.smallest_alignment_index_6d(
                d_x[0, :], d_x[1, :], d_x[2, :], d_x[3, :], d_x[4, :], d_x[5, :],
                d_y[0, :], d_y[1, :], d_y[2, :], d_y[3, :], d_y[4, :], d_y[5, :])
            with h5py.File(hdf5_path, "a") as hdf5_file:
                hdf5_file.create_dataset(
                    f"sali/{time}", data=sali, compression="gzip", shuffle=True)
            

def track_gali_4(p_list: List[xp.Particles], tracker: xt.Tracker, particles_config: ParticlesConfig, run_config: RunConfig, hdf5_path: str, context=xo.ContextCpu()):
    current_time = 0
    d_x = get_displacement_direction(p_list[0], p_list[1], context=context)
    d_px = get_displacement_direction(p_list[0], p_list[2], context=context)
    d_y = get_displacement_direction(p_list[0], p_list[3], context=context)
    d_py = get_displacement_direction(p_list[0], p_list[4], context=context)
    gali = di.global_alignment_index_4_6d(
        d_x[0, :], d_x[1, :], d_x[2, :], d_x[3, :], d_x[4, :], d_x[5, :],
        d_px[0, :], d_px[1, :], d_px[2, :], d_px[3, :], d_px[4, :], d_px[5, :],
        d_y[0, :], d_y[1, :], d_y[2, :], d_y[3, :], d_y[4, :], d_y[5, :],
        d_py[0, :], d_py[1, :], d_py[2, :], d_py[3, :], d_py[4, :], d_py[5, :])
    
    for kind, time in tqdm(run_config.get_event_list()):
        if current_time != time:
            delta_t = time - current_time
            for p in p_list:
                tracker.track(p, num_turns=delta_t)
            current_time = time

        if kind == "normalize":
            for p in p_list[1:]:
                realign_particles(p_list[0], p, run_config.displacement_module, context=context)
        
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
                hdf5_file.create_dataset(
                    f"gali/{time}", data=gali, compression="gzip", shuffle=True)


def track_gali_6(p_list: List[xp.Particles], tracker: xt.Tracker, particles_config: ParticlesConfig, run_config: RunConfig, hdf5_path: str, context=xo.ContextCpu()):
    current_time = 0
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
    
    for kind, time in tqdm(run_config.get_event_list()):
        if current_time != time:
            delta_t = time - current_time
            for p in p_list:
                tracker.track(p, num_turns=delta_t)
            current_time = time

        if kind == "normalize":
            for p in p_list[1:]:
                realign_particles(p_list[0], p, run_config.displacement_module, context=context)
        
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
                hdf5_file.create_dataset(
                    f"gali/{time}", data=gali, compression="gzip", shuffle=True)


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
            hdf5_file.create_dataset(
                f"tunes/{batch}", data=tunes_x, compression="gzip", shuffle=True)
            hdf5_file.create_dataset(
                f"tunes/{batch}", data=tunes_y, compression="gzip", shuffle=True)


class xtrack_engine(object):
    
    def sort_particles(self):
        x = self.context.nparray_from_context_array(self.particles.x)
        px = self.context.nparray_from_context_array(self.particles.px)
        y = self.context.nparray_from_context_array(self.particles.y)
        py = self.context.nparray_from_context_array(self.particles.py)
        zeta = self.context.nparray_from_context_array(self.particles.zeta)
        delta = self.context.nparray_from_context_array(self.particles.delta)
        at_turn = self.context.nparray_from_context_array(
            self.particles.at_turn)
        particle_id = self.context.nparray_from_context_array(
            self.particles.particle_id)

        data = sorted(zip(x, px, y, py, at_turn, zeta, delta, particle_id),
                    key=lambda x: x[7])

        at_turn_data = np.array([x[4] for x in data])

        x_data = np.array([x[0] for x in data])
        x_data[at_turn_data < self.n_turns] = np.nan
        px_data = np.array([x[1] for x in data])
        px_data[at_turn_data < self.n_turns] = np.nan
        y_data = np.array([x[2] for x in data])
        y_data[at_turn_data < self.n_turns] = np.nan
        py_data = np.array([x[3] for x in data])
        py_data[at_turn_data < self.n_turns] = np.nan
        zeta_data = np.array([x[5] for x in data])
        zeta_data[at_turn_data < self.n_turns] = np.nan
        delta_data = np.array([x[6] for x in data])
        delta_data[at_turn_data < self.n_turns] = np.nan

        return ParticlesData(x=x_data, px=px_data, y=y_data, py=py_data, zeta=zeta_data, delta=delta_data, steps=at_turn_data)

    def __init__(self, config: LHCConfig, xy_wall=1.0, context="CPU", device_id="1.0"):
        self.xy_wall = xy_wall
        self.device_id = device_id
        # select context
        if context == "CPU":
            self.context_string = "CPU"
            self.context = xo.ContextCpu(omp_num_threads=os.cpu_count())
        elif context == "CUDA":
            self.context_string = "CUDA"
            self.context = xo.ContextCupy(device=self.device_id)
        elif context == "OPENCL":
            self.context_string = "OPENCL"
            self.context = xo.ContextPyopencl(device=self.device_id)
        else:
            raise ValueError("context not valid")

        # open the line as a json file
        with open(config.mask_path) as f:
            self.line_data = json.load(f)

        # load line
        self.sequence = xt.Line.from_dict(self.line_data)

        # Standard global xy_limits is 1.0 [m]
        # create lattice
        try:
            self.tracker = xt.Tracker(_context=self.context, line=self.sequence, global_xy_limit=self.xy_wall)
        except NameError:
            print("Context not available.")
            print("Switching to CPU context.")
            self.context_string = "CPU"
            self.context = xo.ContextCpu()
            self.tracker = xt.Tracker(_context=self.context, line=self.sequence, global_xy_limit=self.xy_wall)

    def __getstate__(self):
        if hasattr(self, 'n_turns'):
            save_turns = self.n_turns
        else:
            save_turns = 0

        if hasattr(self, 'particles'):
            save_particles = self.particles.copy(_context=xo.ContextCpu()).to_dict()
        else:
            save_particles = None

        return {
            "context": self.context_string,
            "line_data": self.line_data,
            "n_turns": save_turns,
            "particles": save_particles,
            "xy_wall": self.xy_wall,
            "device_id": self.device_id
        }
    
    def __setstate__(self, state):
        self.context_string = state["context"]
        self.line_data = state["line_data"]
        self.n_turns = state["n_turns"]
        self.xy_wall = state["xy_wall"]
        self.device_id = state["device_id"]

        # select context
        if self.context_string == "CPU":
            self.context = xo.ContextCpu()
        elif self.context_string == "CUDA":
            self.context = xo.ContextCupy()
        elif self.context_string == "OPENCL":
            self.context = xo.ContextPyopencl(device=self.device_id)

        # load line
        self.sequence = xt.Line.from_dict(self.line_data)
        
        try:
            # create lattice
            self.tracker = xt.Tracker(
                _context=self.context, line=self.sequence, global_xy_limit=self.xy_wall)
        except NameError:
            print("Required Context not available.")
            print("Switching to CPU context.")
            self.context_string = "CPU"
            self.context = xo.ContextCpu()
            self.tracker = xt.Tracker(
                _context=self.context, line=self.sequence, global_xy_limit=self.xy_wall)

        if state["particles"] is not None:
            self.particles = xp.Particles.from_dict(state["particles"],
                _context=self.context)
        else:
            self.particles = None

    def track(self, x, px, y, py, t, p0c=7000e9, zeta=None, delta=None):
        if zeta is None:
            zeta = np.zeros_like(x)
        if delta is None:
            delta = np.zeros_like(x)
        self.particles = xp.Particles(
            _context=self.context,
            p0c=p0c,
            x=x, px=px, y=y, py=py,
            zeta=zeta, delta=delta)
        self.tracker.track(
            self.particles, num_turns=t, turn_by_turn_monitor=False)
        self.n_turns = t

    def keep_tracking(self, t):
        self.tracker.track(self.particles, num_turns=t,
                           turn_by_turn_monitor=False)
        self.n_turns += t

    def track_and_reverse(self, x, px, y, py, t):
        raise NotImplementedError("Not implemented yet")
        
