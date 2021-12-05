#!/bin/bash
export EOS_MGM_URL=root://eosuser.cern.ch
pwd
ls -alF
nvidia-smi
python3 generic_run.py --input $1 --zeta-option $2 --displacement-kind $3 --time-step advanced --continue-run
eos cp scan*.pkl /eos/project/d/da-and-diffusion-studies/DA_Studies/Simulations/Models/lhc_track_simulations