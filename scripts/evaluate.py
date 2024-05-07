#%%
from log_p_gnn.evaluation import eval_plots, load_from_checkpoint
from log_p_gnn.training import get_loaders, do_mflogp_split
from pathlib import Path
import yaml
import dgl
import torch

#%%

# replace by the path to the checkpoint directory:
checkpoint_dir = 'wandb/latest-run/files/checkpoints'
# checkpoint_dir = 'wandb/run-20240507_165046-3f8kh933/checkpoints'

config_dir = Path(checkpoint_dir).parent/'files/log_p_gnn_config.yaml'

with open(config_dir, 'r') as f:
    config = yaml.safe_load(f)

graphs, _ = dgl.load_graphs(str(config['training']['dspath']))
# %%
# load weights from best checkpoint
model = load_from_checkpoint(checkpoint_dir)

# get dataloaders used for training:
train_graphs, val_graphs, test_graphs = do_mflogp_split(graphs)
train_dl, val_dl, test_dl = get_loaders(train_graphs, val_graphs, test_graphs, config)

# %%
fig, ax = eval_plots(model, train_dl=train_dl, val_dl=val_dl, test_dl=test_dl)
fig.show()
# %%
