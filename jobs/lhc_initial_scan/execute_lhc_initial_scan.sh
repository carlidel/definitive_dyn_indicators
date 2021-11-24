#!/bin/bash
pwd
ls -alF
nvidia-smi
python3 lhc_initial_scan.py --input $1 --short-track