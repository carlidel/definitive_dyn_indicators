# ASK TO GIOVANNI E RICCARDO FOR XSUITE
from henon_map_cpp.dynamic_indicators import abstract_engine
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib

import xobjects as xo
import xtrack as xt

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

class xtrack_engine(abstract_engine):
    def __init__(self, line_path=get_lhc_mask(), xy_wall=1.0, context="CPU", device_id="1.0"):
        # select context
        if context == "CPU":
            self.context_string = "CPU"
            self.context = xo.ContextCpu()
        elif context == "CUDA":
            self.context_string = "CUDA"
            self.context = xo.ContextCupy()
        elif context == "OPENCL":
            self.context_string = "OPENCL"
            self.context = xo.ContextPyopencl(device=device_id)
        else:
            raise ValueError("context not valid")

        # load line
        self.sequence = xt.Line.from_json(line_path)

        # Standard global xy_limits is 1.0 [m]
        # create lattice
        self.tracker = xt.Tracker(_context=self.context, sequence=self.sequence, global_xy_limit=xy_wall)

    def track(self, x, px, y, py, t, p0c=6500e9):
        self.n_turns = 0
        self.particles = xt.Particles(_context=self.context,p0c=p0c, x=x, px=px, y=y, py=py, zeta=np.zeros_like(x), delta=np.zeros_like(x))
        self.tracker.track(self.particles, num_turns=t, turn_by_turn_monitor=False)
        
        if self.context_string == "CUDA" or self.context_string == "OPENCL":
            x = self.particles.x.get()
            px = self.particles.px.get()
            y = self.particles.y.get()
            py = self.particles.py.get()
            at_turn = self.particles.at_turn.get()
            particle_id = self.particles.particle_id.get()
        else:
            x = self.particles.x
            px = self.particles.px
            y = self.particles.y
            py = self.particles.py
            at_turn = self.particles.at_turn
            particle_id = self.particles.particle_id

        data = sorted(zip(x, px, y, py, at_turn, particle_id),
                      key=lambda x: x[5])
        self.n_turns += t

        at_turn_data = np.array([x[4] for x in data])
        x_data = np.array([x[0] for x in data])
        x_data[at_turn_data < self.n_turns] = np.nan
        px_data = np.array([x[1] for x in data])
        px_data[at_turn_data < self.n_turns] = np.nan
        y_data = np.array([x[2] for x in data])
        y_data[at_turn_data < self.n_turns] = np.nan
        py_data = np.array([x[3] for x in data])
        py_data[at_turn_data < self.n_turns] = np.nan

        return x_data, px_data, y_data, py_data, at_turn_data
    
    def keep_tracking(self, t):
        self.tracker.track(self.particles, num_turns=t,
                           turn_by_turn_monitor=False)

        if self.context_string == "CUDA" or self.context_string == "OPENCL":
            x = self.particles.x.get()
            px = self.particles.px.get()
            y = self.particles.y.get()
            py = self.particles.py.get()
            at_turn = self.particles.at_turn.get()
            particle_id = self.particles.particle_id.get()
        else:
            x = self.particles.x
            px = self.particles.px
            y = self.particles.y
            py = self.particles.py
            at_turn = self.particles.at_turn
            particle_id = self.particles.particle_id

        data = sorted(zip(x, px, y, py, at_turn, particle_id),
                      key=lambda x: x[5])
        self.n_turns += t

        at_turn_data = np.array([x[4] for x in data])
        x_data = np.array([x[0] for x in data])
        x_data[at_turn_data < self.n_turns] = np.nan
        px_data = np.array([x[1] for x in data])
        px_data[at_turn_data < self.n_turns] = np.nan
        y_data = np.array([x[2] for x in data])
        y_data[at_turn_data < self.n_turns] = np.nan
        py_data = np.array([x[3] for x in data])
        py_data[at_turn_data < self.n_turns] = np.nan

        return x_data, px_data, y_data, py_data, at_turn_data

    def track_and_reverse(self, x, px, y, py, t):
        raise NotImplementedError("Not implemented yet")
        
