#!/bin/bash
python3 henon_run.py --omega-x $1 --omega-y $2 --epsilon $3 --mu $4 --kick-module $5 --kick-sigma $6 --modulation-kind $7 --omega-0 $8 -d $9 --tracking ${10}

echo "Executed: python3 henon_run.py --omega-x $1 --omega-y $2 --epsilon $3 --mu $4 --kick-module $5 --kick-sigma $6 --modulation-kind $7 --omega-0 $8 -d $9 --tracking ${10}"

echo "Moving files to the output directory..."

xrdcp *.hdf5 root://eosuser.cern.ch//eos/project/d/da-and-diffusion-studies/DA_Studies/Simulations/Models/dynamic_indicator_analysis/big_data/

echo "Moved files to the output directory!"

rm -v *.hdf5

echo "Removed files!"