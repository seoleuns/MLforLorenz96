#!/bin/bash

# Build numerical model
pushd three-lev
make clean
make
popd

# Add all modules to the Python path
PWD=`pwd`
export PYTHONPATH=$PYTHONPATH:$PWD

