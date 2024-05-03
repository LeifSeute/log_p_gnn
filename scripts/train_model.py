#%%
from log_p_gnn.grappa_model import GrappaGNN
from log_p_gnn.readout import Readout, Denormalize
from log_p_gnn.dgl_pooling import DGLPooling
from sklearn.model_selection import train_test_split
from pathlib import Path

import torch
torch.set_float32_matmul_precision('medium')
import dgl

dspath = '../data/dgl_dataset.bin'
# dspath = '../data/all_atom_dgl_dataset.bin'

graphs, _ = dgl.load_graphs(dspath)

g = dgl.batch(graphs[:10])

#%%
exclude_feats = ['logp']

feat_names = []
feat_dims = {}
for k in g.nodes['atom'].data.keys():
    v = g.nodes['atom'].data[k]
    if not k in exclude_feats:
        feat_names.append(k)
        feat_dims[k] = v.shape[-1] if len(v.shape) > 1 else 1

print('using features:')
print(feat_dims)
#%%


config = {
    'feats': 64,
    'in_feat_name': feat_names,     
    'in_feat_dims': feat_dims,      
    'n_att': 1,
    'n_conv': 0,
    'n_heads': 8,
    'attention_dropout': 0.,
    'final_dropout': 0.,
    'initial_dropout': 0.,
    'out_feats': 12,
    'pooling': {
        'max': False,
        'mean': False,
        'sum': True
    },
    'readout': {
        'num_layers': 2,
        'dropout': 0.,
        'feats': 12
    },
    'denormalizer': {
        'learnable': False
    }
}


def get_model(config, train_target_mean=0, train_target_std=1):

    gnn = GrappaGNN(out_feats=config['out_feats'], in_feat_name=config['in_feat_name'], in_feat_dims=config['in_feat_dims'], n_att=config['n_att'], n_heads=config['n_heads'], attention_dropout=config['attention_dropout'], final_dropout=config['final_dropout'], charge_encoding=False, n_conv=config['n_conv'], initial_dropout=config['initial_dropout'], layer_norm=True)

    pooler = DGLPooling(**config['pooling'])

    pooling_out_dim = sum([config['out_feats'] for _ in config['pooling'].values() if _])

    readout = Readout(in_features=pooling_out_dim, out_features=1, hidden_features=config['readout']['feats'], num_layers=config['readout']['num_layers'], dropout=config['readout']['dropout'])

    denormalizer = Denormalize(mean=train_target_mean, std=train_target_std, learnable=config['denormalizer']['learnable'])

    model = torch.nn.Sequential(gnn, pooler, readout, denormalizer)

    return model


# %%

    
# SPLIT THE DATA ACCORDING TO MFPLOGP:
x = graphs
y = [None for _ in x] # our targets are stored in the graphs themselves

# copied from https://github.com/TeixeiraResearchLab/MF-LOGP_Development-/blob/main/MFLOGP_Training_Script.py:
##################################################################################
[X, Vault_X, Y, Vault_Y] = train_test_split(x,y,train_size = 0.85, random_state = 42)
[X_train,X_test,y_train,y_test]=train_test_split(X,Y,train_size=0.8,shuffle = True)
##################################################################################

# rename:
train_graphs = X_train
val_graphs = X_test # this is actually used for validation
test_graphs = Vault_X # this is the completely hold-out test set

print('number of training graphs:', len(train_graphs))
print('number of validation graphs:', len(val_graphs))
print('number of test graphs:', len(test_graphs))

#%%

# normalize the targets using the training set:

train_targets = torch.cat([g.nodes['global'].data['logp'] for g in train_graphs], dim=0)

mean = train_targets.mean()
std = train_targets.std()

model = get_model(config, train_target_mean=mean, train_target_std=std)
#%%
from log_p_gnn.data_utils import DGLDataset
from torch.utils.data import DataLoader

# to speed up training, repeat the training set
# (there is some overhead each epoch)
train_graphs = [g for g in train_graphs for _ in range(10)]


train_ds, val_ds, test_ds = DGLDataset(graphs=train_graphs), DGLDataset(graphs=val_graphs), DGLDataset(graphs=test_graphs)


batch_size = 32 if len(graphs) < 10000 else 128

train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=train_ds.collate_fn, drop_last=True, num_workers=2)

val_dl = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False, collate_fn=val_ds.collate_fn, num_workers=1)

test_dl = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False, collate_fn=test_ds.collate_fn, num_workers=1)

#%%

from torch.optim import Adam
import pytorch_lightning as pl

class GrappaGNNModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = model
        self.loss = torch.nn.MSELoss()
        self.opt = Adam(model.parameters(), lr=2e-5)
        self.noise_level = 0.2
        
    def forward(self, g):
        return self.model(g)[:,0]
    
    def training_step(self, batch, batch_idx):
        g = batch
        batchsize = g.num_nodes('global')
        x = self.forward(g)
        target = g.nodes['global'].data['logp']
        loss = self.loss(x, target+torch.randn_like(target)*self.noise_level)
        self.log('train_loss', loss, batch_size=batchsize, on_step=False, on_epoch=True)
        self.log('train_rmse', torch.sqrt(loss), batch_size=batchsize, on_step=False, on_epoch=True)
        self.log('train_std_dev', target.std(), batch_size=batchsize, on_step=False, on_epoch=True)
        self.log('batch_size', batchsize, batch_size=batchsize)
        return loss
    
    def validation_step(self, batch, batch_idx):
        g = batch
        batchsize = g.num_nodes('global')
        x = self.forward(g)
        target = g.nodes['global'].data['logp']
        rmse = torch.sqrt(torch.square(x - target).mean())
        mae = torch.abs(x - target).mean()
        self.log('val_rmse', rmse, batch_size=batchsize, on_step=False, on_epoch=True)
        self.log('val_mae', mae, batch_size=batchsize, on_step=False, on_epoch=True)
        self.log('val_std_dev', target.std(), batch_size=batchsize, on_step=False, on_epoch=True)
        self.log('batch_size', batchsize, batch_size=batchsize, on_step=False, on_epoch=True)
    
    def test_step(self, batch, batch_idx):
        g = batch
        batchsize = g.num_nodes('global')
        x = self.forward(g)
        target = g.nodes['global'].data['logp']
        rmse = torch.sqrt(torch.square(x - target).mean())
        mae = torch.abs(x - target).mean()

        # Optionally, add metrics to Lightning's logs without creating a plot in WandB
        wandb.run.summary["final_test_rmse"] = rmse.item()
        wandb.run.summary["final_test_mae"] = mae.item()
        wandb.run.summary["test_std_dev"] = target.std().item()

    
    def configure_optimizers(self):
        return self.opt

#%%

import wandb

wandb.finish()
wandb.init(project='logp-gnn')

logger = pl.loggers.WandbLogger()

wandb_path = Path(wandb.run.dir)

from pytorch_lightning.callbacks import ModelCheckpoint

# Create a ModelCheckpoint callback that saves the two best models based on the lowest validation RMSE
checkpoint_callback = ModelCheckpoint(
    monitor='val_rmse',
    dirpath=str(wandb_path.parent / 'checkpoints'),
    filename='best-model-{epoch:02d}-{val_rmse:.2f}',
    save_top_k=2,
    mode='min',
    save_weights_only=True,
    save_on_train_epoch_end=False,  # Only save on validation end
    every_n_epochs=10,  # Save every 50 epochs
    save_last=True
)

# save the model config as yaml:
import yaml

with open(wandb_path / 'model_config.yaml', 'w') as f:
    yaml.dump(config, f)

#%%

module = GrappaGNNModule()

# onlycalculate metrics on validation set every 10 epochs:
trainer = pl.Trainer(max_epochs=5000, accelerator='gpu', logger=logger, callbacks=[checkpoint_callback], check_val_every_n_epoch=1, log_every_n_steps=10, gradient_clip_val=1.0)

trainer.fit(module, train_dl, val_dl)
# %%
print('\n\nEvaluating model with best validation RMSE...')
print(f'Best model path: {checkpoint_callback.best_model_path}\n')
best_model_path = checkpoint_callback.best_model_path
module = GrappaGNNModule.load_from_checkpoint(best_model_path)

# Perform testing with the best model
trainer.test(module, test_dl)
# %%

# create a plot of the model's predictions vs. ground truth on the test set:

import torch
from pathlib import Path

import matplotlib.pyplot as plt

FONTSIZE = 22

plt.rcParams.update({'font.size': FONTSIZE})
plt.rcParams.update({'axes.labelsize': FONTSIZE})
plt.rcParams.update({'xtick.labelsize': FONTSIZE})
plt.rcParams.update({'ytick.labelsize': FONTSIZE})
plt.rcParams.update({'legend.fontsize': FONTSIZE})


checkpoint_dir = wandb_path.parent/'checkpoints'
# checkpoint_dir = 'wandb/run-20240502_180246-w82pllm1/checkpoints'

ckpt_paths = list(Path(checkpoint_dir).glob('*.ckpt'))

# pick the path with the smallest validation RMSE:
val_rmses = [float(str(p).split('val_rmse=')[-1].replace('.ckpt', '')) if 'val_rmse' in str(p) else float('inf') for p in ckpt_paths]
best_ckpt = ckpt_paths[val_rmses.index(min(val_rmses))]


# %%
# load the model:
import yaml
config = yaml.load((Path(checkpoint_dir).parent / 'files/model_config.yaml').read_text(), Loader=yaml.FullLoader)
model = get_model(config)

# load the weights:
state_dict = torch.load(best_ckpt)['state_dict']

# remove the trainling 'model.' from the keys:
state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}

model.load_state_dict(state_dict)

#%%

# plot train, val and test scatter plots:
import numpy as np
import seaborn as sns

fig, ax = plt.subplots(2, 3, figsize=(15, 10))

for i, (dl, title) in enumerate(zip([train_dl, val_dl, test_dl], ['Train', 'Validation', 'Test'])):
    predictions, targets = [], []
    for g in dl:
        with torch.no_grad():
            model = model.eval()
            model = model.to('cuda')
            g = g.to('cuda')
            x = model(g)[:,0].detach().cpu().numpy()
            y = g.nodes['global'].data['logp'].detach().cpu().numpy()
        predictions.append(x)
        targets.append(y)


    x = np.concatenate(predictions)
    y = np.concatenate(targets)
    rmse = np.sqrt(np.mean((x - y)**2))

    ax[0][i].scatter(y, x, s=1)
    min_val = min(min(y), min(x))
    max_val = max(max(y), max(x))
    ax[0][i].plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1)
    ax[0][i].set_title(title)
    ax[0][i].set_xlabel('Ground truth')
    ax[0][i].set_ylabel('Prediction')
    ax[0][i].text(0.05, 0.9, f'RMSE: {rmse:.2f}', transform=ax[0][i].transAxes)

    # seaborn density plot:
    sns.kdeplot(y=x, x=y, ax=ax[1][i], cmap='viridis', fill=True, thresh=0.1, levels=20, cbar=False)

    # set the same limits as above:
    ax[1][i].set_xlim(min_val, max_val)
    ax[1][i].set_ylim(min_val, max_val)
    ax[1][i].plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1)
    ax[1][i].set_xlabel('Ground truth')
    ax[1][i].set_ylabel('Prediction') if i == 0 else None

fig.savefig(Path(checkpoint_dir).parent / 'scatter_plots.png', dpi=300)
# plt.show()
# %%
