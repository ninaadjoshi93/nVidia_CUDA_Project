#!/bin/bash
# This script is used to run the heat distribution example
# The prerequisite is to load the cudatoolkit module into the environment
# and use a node with an nVidia GPU
module load cudatoolkit
make run
