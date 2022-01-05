# ASK TO GIOVANNI E RICCARDO FOR XSUITE
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
import json

import xobjects as xo
from xobjects import context_cpu
import xtrack as xt
import xpart as xp

from cpymad.madx import Madx


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

        return dict(x=x_data, px=px_data, y=y_data, py=py_data, zeta=zeta_data, delta=delta_data, steps=at_turn_data)

    def __init__(self, line_path=get_lhc_mask(), xy_wall=1.0, context="CPU", device_id="1.0"):
        self.xy_wall = xy_wall
        self.device_id = device_id
        # select context
        if context == "CPU":
            self.context_string = "CPU"
            self.context = xo.ContextCpu()
        elif context == "CUDA":
            self.context_string = "CUDA"
            self.context = xo.ContextCupy()
        elif context == "OPENCL":
            self.context_string = "OPENCL"
            self.context = xo.ContextPyopencl(device=self.device_id)
        else:
            raise ValueError("context not valid")

        # open the line as a json file
        with open(line_path) as f:
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
        
