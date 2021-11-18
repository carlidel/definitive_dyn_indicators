#!/bin/bash

# execute nvidia-smi to get the GPU information and pipe it to a file
nvidia-smi | tee nvidia-smi.txt

# execute nvidia-smi again to get the GPU information and pipe it to stderr
nvidia-smi 2>&1

# execute nvidia-smi again to get the GPU information and pipe it to stdout
nvidia-smi >&1