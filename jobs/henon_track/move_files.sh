#!/bin/bash

export EOS_MGM_URL=root://eosuser.cern.ch
eos cp /afs/cern.ch/work/c/camontan/public/definitive_dyn_indicators/data/$1 /eos/project/d/da-and-diffusion-studies/DA_Studies/Simulations/Models/dynamic_indicator_analysis/big_data/
rm -v /afs/cern.ch/work/c/camontan/public/definitive_dyn_indicators/data/$1