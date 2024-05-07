from log_p_gnn.get_model import get_model
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import List, Tuple
import wandb
from copy import deepcopy
import yaml

import torch
torch.set_float32_matmul_precision('medium')
import dgl
from log_p_gnn.data_utils import DGLDataset
from torch.utils.data import DataLoader

from torch.optim import Adam
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from log_p_gnn.evaluation import get_best_checkpoint_path



def train_model(config:dict):
    graphs, _ = dgl.load_graphs(str(config['training']['dspath']))
    config['gnn']['in_feat_dims'] = infer_feat_dims(graphs[0], config['gnn']['in_feat_name'])

    # get dataloaders:
    train_graphs, val_graphs, test_graphs = do_mflogp_split(graphs)
    train_dl, val_dl, test_dl = get_loaders(train_graphs, val_graphs, test_graphs, config)

    # init the model
    # use the training set to normalize the targets:
    train_targets = torch.cat([g.nodes['global'].data['logp'].flatten() for g in train_dl], dim=0)
    mean = train_targets.mean()
    std = train_targets.std()

    model = get_model(config, train_target_mean=mean, train_target_std=std)

    # test call to trigger exception before starting training:
    dummy_graph = deepcopy(train_graphs[0])
    model(dummy_graph)

    # init the trainer
    trainer, wandb_path = get_trainer(config)

    module = GrappaGNNModule(model, config)

    # start the training
    trainer.fit(module, train_dl, val_dl)

    # Evaluate the best-validation model on the test set
    print('\n\nEvaluating model with best validation RMSE...')
    best_model_path = get_best_checkpoint_path(wandb_path.parent/'checkpoints')
    print(f'Best model path: {best_model_path}\n')
    state_dict = torch.load(best_model_path)['state_dict']
    module.load_state_dict(state_dict)

    # Perform testing with the best model
    trainer.test(module, test_dl)




def get_trainer(config)->pl.Trainer:
    wandb.finish()
    wandb.init(project=config['training']['project'], name=config['training']['name'])

    logger = pl.loggers.WandbLogger()
    wandb_path = Path(wandb.run.dir)

    # add config to wandb:
    wandb.config.update(config)

    # Create a ModelCheckpoint callback that saves the two best models based on the lowest validation RMSE
    checkpoint_callback = ModelCheckpoint(
        monitor='val_rmse',
        dirpath=str(wandb_path.parent / 'checkpoints'),
        filename='best-model-{epoch:02d}-{val_rmse:.2f}',
        save_top_k=2,
        mode='min',
        save_weights_only=True,
        save_on_train_epoch_end=False,  # Only save on validation end
        every_n_epochs=10,  # Save every 10 epochs
        save_last=True
    )

    # save the model config as yaml:
    with open(wandb_path / 'log_p_gnn_config.yaml', 'w') as f:
        yaml.dump(config, f)

    trainer = pl.Trainer(max_epochs=2000, accelerator='gpu', logger=logger, callbacks=[checkpoint_callback], check_val_every_n_epoch=1, log_every_n_steps=1, gradient_clip_val=1.0)

    return trainer, wandb_path
    
class GrappaGNNModule(pl.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.loss = torch.nn.MSELoss()
        self.opt = Adam(model.parameters(), lr=config['training']['lr'])
        self.noise_level = config['training']['noise_level']

        self.accumulated_batches = 0
        self.epoch_cycles = config['training']['epoch_cycles']

        self.test_step_counter = 0
        
    def forward(self, g):
        return self.model(g)[:,0]        

    def training_step(self, batch, batch_idx):
        g = batch
        batchsize = len(g.nodes['global'].data['logp'].flatten())
        x = self.forward(g)
        target = g.nodes['global'].data['logp']

        # add noise to the target:
        loss = self.loss(x, target+torch.randn_like(target)*self.noise_level)
        
        # self.log('train_loss', loss, batch_size=batchsize, on_step=False, on_epoch=True)
        self.log('train_rmse', torch.sqrt(loss), batch_size=batchsize, on_step=False, on_epoch=True)
        # self.log('train_std_dev', target.std(), batch_size=batchsize, on_step=False, on_epoch=True)
        # self.log('batch_size', batchsize, batch_size=batchsize)
        return loss
    
    def validation_step(self, batch, batch_idx):
        g = batch
        batchsize = len(g.nodes['global'].data['logp'].flatten())

        x = self.forward(g)
        target = g.nodes['global'].data['logp']
        rmse = torch.sqrt(torch.square(x - target).mean())
        mae = torch.abs(x - target).mean()
        self.log('val_rmse', rmse, batch_size=batchsize, on_step=False, on_epoch=True)
        self.log('val_mae', mae, batch_size=batchsize, on_step=False, on_epoch=True)
        # self.log('val_std_dev', target.std(), batch_size=batchsize, on_step=False, on_epoch=True)
        self.log('batch_size', batchsize, batch_size=batchsize, on_step=False, on_epoch=True)
    
    def test_step(self, batch, batch_idx):
        g = batch
        x = self.forward(g)
        target = g.nodes['global'].data['logp']
        rmse = torch.sqrt(torch.square(x - target).mean())
        mae = torch.abs(x - target).mean()

        # Optionally, add metrics to Lightning's logs without creating a plot in WandB
        wandb.run.summary["final_test_rmse"] = rmse.item()
        wandb.run.summary["final_test_mae"] = mae.item()
        wandb.run.summary["test_std_dev"] = target.std().item()

        if self.test_step_counter > 1:
            raise ValueError('test_step was called more than once, number of batches on the test dataloader must be set to 1')
        self.test_step_counter += 1

    def configure_optimizers(self):
        return self.opt


def infer_feat_dims(g:dgl.DGLGraph, feat_names:list):
    feat_dims = {}
    for k in feat_names:
        if not k in g.nodes['atom'].data.keys():
            raise ValueError(f'{k} not in graph features: {g.nodes["atom"].data.keys()}')
        v = g.nodes['atom'].data[k]
        feat_dims[k] = v.shape[-1] if len(v.shape) > 1 else 1
    return feat_dims


def do_mflogp_split(graphs:List[dgl.DGLGraph])->Tuple[List[dgl.DGLGraph], List[dgl.DGLGraph], List[dgl.DGLGraph]]:
    
    # SPLIT THE DATA ACCORDING TO MFPLOGP:
    x = graphs
    y = [None for _ in x] # our targets are stored in the graphs themselves

    # copied from https://github.com/TeixeiraResearchLab/MF-LOGP_Development-/blob/main/MFLOGP_Training_Script.py:
    ##################################################################################
    [X, Vault_X, Y, Vault_Y] = train_test_split(x,y,train_size = 0.85, random_state = 42)
    [X_train,X_test,y_train,y_test]=train_test_split(X,Y,train_size=0.8,shuffle = True, random_state=42)
    ##################################################################################

    # rename:
    train_graphs = X_train
    val_graphs = X_test # this is actually used for validation
    test_graphs = Vault_X # this is the completely hold-out test set

    print('number of training graphs:', len(train_graphs))
    print('number of validation graphs:', len(val_graphs))
    print('number of test graphs:', len(test_graphs))

    std_val = torch.cat([g.nodes['global'].data['logp'].flatten() for g in val_graphs]).std()
    std_test = torch.cat([g.nodes['global'].data['logp'].flatten() for g in test_graphs]).std()
    print('std of validation set:', std_val)
    print('std of test set:', std_test)

    return train_graphs, val_graphs, test_graphs


def get_loaders(train_graphs:List[dgl.DGLGraph], val_graphs:List[dgl.DGLGraph], test_graphs:List[dgl.DGLGraph], config):

    # make the train set num_cycles times larger:
    train_graphs = [g for g in train_graphs for _ in range(config['training']['epoch_cycles'])]

    train_ds, val_ds, test_ds = DGLDataset(graphs=train_graphs), DGLDataset(graphs=val_graphs), DGLDataset(graphs=test_graphs)

    train_dl = DataLoader(train_ds, batch_size=config['training']['batch_size'], shuffle=True, collate_fn=train_ds.collate_fn, drop_last=True, num_workers=2)

    val_dl = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False, collate_fn=val_ds.collate_fn, num_workers=1)

    test_dl = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False, collate_fn=test_ds.collate_fn, num_workers=1)

    return train_dl, val_dl, test_dl