#!/bin/bash
pwd
ls -alF
nvidia-smi
python3 generic_run.py --input $1 --zeta-option $2 --displacement-kind $3 --time-step advanced