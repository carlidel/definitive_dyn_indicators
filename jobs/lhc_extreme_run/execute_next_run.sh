#!/bin/bash
pwd
ls -alF
echo $1  >> /afs/cern.ch/work/c/camontan/public/definitive_dyn_indicators/jobs/lhc_extreme_run/log/log/nvidia-smi.txt
echo $2  >> /afs/cern.ch/work/c/camontan/public/definitive_dyn_indicators/jobs/lhc_extreme_run/log/log/nvidia-smi.txt
nvidia-smi >> /afs/cern.ch/work/c/camontan/public/definitive_dyn_indicators/jobs/lhc_extreme_run/log/log/nvidia-smi.txt
python3 generic_run.py --input $1 --zeta-option $2 --displacement-kind none --time-step basic --continue-run