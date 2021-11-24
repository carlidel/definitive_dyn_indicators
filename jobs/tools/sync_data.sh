#!/bin/bash
# source dir
SOURCE_DIR=/afs/cern.ch/work/c/camontan/public/definitive_dyn_indicators/data

# get directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# get parent directory
DIR="$(dirname $DIR)"
# get grandparent directory
DIR="$(dirname $DIR)"

# define output dir
OUTPUT_DIR=$DIR/data

# sync data
rsync -avz --progress $SOURCE_DIR/ $OUTPUT_DIR