#!/bin/bash
export EOS_MGM_URL=root://eosuser.cern.ch

eos cp /afs/cern.ch/work/c/camontan/public/definitive_dyn_indicators/data/scan*basic*.pkl /eos/project/d/da-and-diffusion-studies/DA_Studies/Simulations/Models/lhc_track_simulations

rm -v /afs/cern.ch/work/c/camontan/public/definitive_dyn_indicators/data/scan*basic*nturns*.pkl