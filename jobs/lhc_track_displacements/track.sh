#!/bin/bash
python3 generic_run.py --input $1 -z $2 -t $3 

echo "Executed $1, $2, $3"

echo "Moving files to the output directory..."

xrdcp *.hdf5 root://eosuser.cern.ch//eos/project/d/da-and-diffusion-studies/DA_Studies/Simulations/Models/dynamic_indicator_analysis/big_data/ --force

xrdcp *.pkl root://eosuser.cern.ch//eos/project/d/da-and-diffusion-studies/DA_Studies/Simulations/Models/dynamic_indicator_analysis/big_data/ --force

echo "Moved files to the output directory!"

rm -v *.hdf5
rm -v *.pkl

echo "Removed files!"