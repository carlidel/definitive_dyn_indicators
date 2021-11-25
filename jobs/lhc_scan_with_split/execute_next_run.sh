#!/bin/bash
pwd
ls -alF
nvidia-smi
python3 generic_run.py --input $1 --continue-run