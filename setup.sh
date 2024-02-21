#!/bin/bash

# bash tuto directory of the script when is sourced
# https://www.baeldung.com/linux/bash-get-location-within-script
rshower_package=$(realpath $(dirname ${BASH_SOURCE}))


export PYTHONPATH=$rshower_package/src:$PYTHONPATH
export PATH=$rshower_package/src/scripts:$PATH
