#!/bin/bash
pwd
ls -alF

python3 run.py --hdf5_filename $1 --checkpoint_filename $2 --hl_lhc $3 --particle_config $4 --dyn_ind $5 --context gpu