cuda_version=${1:-"cpu"}

# check that we are not in base environment:
if [ "$CONDA_DEFAULT_ENV" == "base" ]; then
    echo "Please activate a conda environment before running this script."
    exit 1
fi

echo "Installing log_p_gnn in environment: $CONDA_DEFAULT_ENV"

# if not cpu mode:
# conda install python=3.10 pytorch=2.1.0 dgl -c pytorch -c dglteam -y


# install for cuda 11.8 (other versions usually make problems with dgl...)
conda install python=3.10 pytorch=2.1.0 pytorch-cuda=11.8 dgl -c pytorch -c nvidia -c dglteam/label/cu118 -y

pip install -r requirements.txt
pip install -e .

set -e .
python test_install.py
echo "Installation successful. Environment name: $CONDA_DEFAULT_ENV."