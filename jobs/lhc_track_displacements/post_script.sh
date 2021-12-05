#!/bin/bash
export EOS_MGM_URL=root://eosuser.cern.ch

eos cp $1 /eos/project/d/da-and-diffusion-studies/DA_Studies/Simulations/Models/lhc_track_simulations

rm -v $1