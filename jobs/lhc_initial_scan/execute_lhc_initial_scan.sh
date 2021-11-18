#!/bin/bash

export EOS_MGM_URL=root://eosuser.cern.ch
export MYPYTHON=/afs/cern.ch/work/c/camontan/public/anaconda3

unset PYTHONHOME
unset PYTHONPATH
source $MYPYTHON/bin/activate
export PATH=$MYPYTHON/bin:$PATH

source /afs/cern.ch/work/c/camontan/public/definitive_dyn_indicators/venv/bin/activate

which python

python3 lhc_initial_scan.py $1