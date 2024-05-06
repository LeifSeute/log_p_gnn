#!/bin/bash

# Default to "cpu" if no argument provided, otherwise use the provided CUDA version
cuda_version=${1:-"cpu"}

# Check that we are not in the base Conda environment
if [ "$CONDA_DEFAULT_ENV" == "base" ]; then
    echo "Please activate a Conda environment before running this script."
    exit 1
fi

echo "Installing log_p_gnn in environment: $CONDA_DEFAULT_ENV"

# If not in CPU mode, install with CUDA support
if [ "$cuda_version" != "cpu" ]; then
    conda install python=3.10 pytorch=2.1.0 dgl -c pytorch -c dglteam -y
else
    # install for cuda 11.8 (other versions usually make problems with dgl...)
    conda install python=3.10 pytorch=2.1.0 pytorch-cuda=11.8 dgl -c pytorch -c nvidia -c dglteam/label/cu118 -y
fi

pip install -r requirements.txt
pip install -e .

set -e .
python test_install.py
echo "Installation successful. Environment name: $CONDA_DEFAULT_ENV."