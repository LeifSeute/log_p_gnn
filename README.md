### Installation

```
conda create -n log_p_gnn -y
conda activate log_p_gnn
bash installation.sh
# install CGsmiles for creating datasets
```


### Dataset Creation
```
bash scripts/generate_datasets.sh
```

### Training
```
cd scripts
python train_cg.py # (You will need to login to your free wandb account for tracking the metrics. Modify the config dict in the script to change the hyperparameters)
```