
THISDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
pushd $THISDIR

python create_cg_all_atom_dataset.py
python create_cg_dataset.py
python create_all_atom_dataset.py

popd