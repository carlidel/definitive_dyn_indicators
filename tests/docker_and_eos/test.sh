#!/bin/bash

# create a file of 100MB with random data
dd if=/dev/urandom of=test.dat bs=1M count=100

# copy to eos with xrdcp
xrdcp test.dat root://eosuser.cern.ch//eos/project/d/da-and-diffusion-studies/DA_Studies/Simulations/Models/dynamic_indicator_analysis/big_data/test.dat