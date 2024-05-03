#%%
from log_p_gnn.grappa_model import GrappaGNN
from log_p_gnn.readout import Readout, Denormalize
from log_p_gnn.dgl_pooling import DGLPooling
from sklearn.model_selection import train_test_split
from pathlib import Path

import torch
torch.set_float32_matmul_precision('medium')
import dgl

dspath = '../data/all_atom_dgl_dataset.bin'

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

# feat_dims['atomic_number'] = len(ELEMENTS)

print(feat_names)
print(feat_dims)
#%%

config = {
    'feats': 256,
    'in_feat_name': feat_names,     
    'in_feat_dims': feat_dims,      
    'n_att': 8,
    'n_heads': 8,
    'attention_dropout': 0.4,
    'final_dropout': 0.4,
    'pooling': {
        'max': True,
        'mean': True,
        'sum': True
    },
    'readout': {
        'num_layers': 6,
        'dropout': 0.4
    },
    'denormalizer': {
        'learnable': True
    }
}
# feats = 256

# gnn = GrappaGNN(feats, feat_names, feat_dims, n_att=8, n_heads=8, attention_dropout=0.4, final_dropout=0.4)

gnn = GrappaGNN(out_feats=config['feats'], in_feat_name=config['in_feat_name'], in_feat_dims=config['in_feat_dims'], n_att=config['n_att'], n_heads=config['n_heads'], attention_dropout=config['attention_dropout'], final_dropout=config['final_dropout'], charge_encoding=False)

g = gnn(g)
# %%

pooler = DGLPooling(max=True, mean=True, sum=True)
pooler = DGLPooling(**config['pooling'])

x = pooler(g)
x.shape
#%%
# readout = Readout(in_features=3*feats, out_features=1, hidden_features=feats, num_layers=6, dropout=0.4)
readout = Readout(in_features=3*config['feats'], out_features=1, hidden_features=config['feats'], num_layers=config['readout']['num_layers'], dropout=config['readout']['dropout'])

model = torch.nn.Sequential(gnn, pooler, readout)

# %%

x = model(g)
x.shape
# %%
loss = torch.nn.MSELoss()


target = g.nodes['global'].data['logp']

x = x[:, 0]
loss(x, target)
# %%
# not so much normalization needed
# %%
from torch.utils.data import DataLoader, Dataset

class DGLDataset(Dataset):
    def __init__(self, dspath:str=None, graphs:list=None):
        assert not (dspath is None and graphs is None), 'Either dspath or graphs must be provided'
        assert not (dspath is not None and graphs is not None), 'Only one of dspath or graphs must be provided'

        if graphs is not None:
            self.graphs = graphs
        else:    
            self.graphs, _ = dgl.load_graphs(str(dspath))
        
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        return self.graphs[idx]
    
    def collate_fn(self, batch):
        return dgl.batch(batch)
    
# ds = DGLDataset(dspath)
# split into train val test:
# train_size = int(0.8 * len(ds))
# val_size = int(0.1 * len(ds))
# test_size = len(ds) - train_size - val_size

# train_ds, val_ds, test_ds = torch.utils.data.random_split(ds, [train_size, val_size, test_size])

#%%

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

#%%

# normalize the targets using the training set:

train_targets = torch.cat([g.nodes['global'].data['logp'] for g in train_graphs], dim=0)


mean = train_targets.mean()
std = train_targets.std()

full_model = torch.nn.Sequential(gnn, pooler, readout, Denormalize(mean, std, learnable=True))
#%%

train_ds, val_ds, test_ds = DGLDataset(graphs=train_graphs), DGLDataset(graphs=val_graphs), DGLDataset(graphs=test_graphs)

batch_size = 32 if len(graphs) < 200 else 256

train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=train_ds.collate_fn, drop_last=True, num_workers=2)

val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=val_ds.collate_fn, num_workers=1)

test_dl = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False, collate_fn=test_ds.collate_fn, num_workers=1)

#%%

from torch.optim import Adam

opt = Adam(full_model.parameters(), lr=1e-5)

#%%

import pytorch_lightning as pl

class GrappaGNNModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = full_model
        self.loss = loss
        self.opt = opt
        
    def forward(self, g):
        return self.model(g)[:,0]
    
    def training_step(self, batch, batch_idx):
        g = batch
        batchsize = g.num_nodes('global')
        x = self.forward(g)
        target = g.nodes['global'].data['logp']
        loss = self.loss(x, target)
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
        return loss
    
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
    every_n_epochs=5,  # Save every 50 epochs
    save_last=True
)
#%%

module = GrappaGNNModule()

# onlycalculate metrics on validation set every 10 epochs:
trainer = pl.Trainer(max_epochs=5000, accelerator='gpu', logger=logger, callbacks=[checkpoint_callback], check_val_every_n_epoch=5, log_every_n_steps=1, gradient_clip_val=1.0)

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

# checkpoint_dir = wandb_path.parent
checkpoint_dir = 'wandb/run-20240502_152744-i3weoyks/checkpoints/'

ckpt_paths = list(Path(checkpoint_dir).glob('*.ckpt'))

# pick the path with the smallest validation RMSE:
val_rmses = [float(str(p).split('val_rmse=')[-1].replace('.ckpt', '')) if 'val_rmse' in str(p) else float('inf') for p in ckpt_paths]
best_ckpt = ckpt_paths[val_rmses.index(min(val_rmses))]


# %%
# load the model:
model = torch.nn.Sequential(gnn, pooler, readout)
full_model = torch.nn.Sequential(model, Denormalize())



# %%
