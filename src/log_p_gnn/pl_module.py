import torch
import pytorch_lightning as pl
import wandb
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import importlib

def _get_model_class(class_path: str):
    # Split the module path and class name
    module_name, class_name = class_path.rsplit('.', 1)
    # Dynamically import the module
    module = importlib.import_module(module_name)
    # Get the class from the module
    return getattr(module, class_name)


class PLModule(pl.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.cfg = config.experiment.training
        self.delta = config.experiment.training.delta
        self.supress_log = False

    def loss_fn(self, x, target):
        # only apply mse where target is not nan
        mask = ~torch.isnan(target)
        t = target[mask]
        x_ = x[mask]
        return torch.nn.functional.huber_loss(x_, t, delta=self.delta)

    def l1_l2_loss(self, x, target):
        # only apply mse where target is not nan
        mask = ~torch.isnan(target)
        t = target[mask]
        x_ = x[mask]
        return torch.nn.functional.l1_loss(x_, t), torch.nn.functional.mse_loss(x_, t)


    def forward(self, g):
        return self.model(g)

    def training_step(self, batch, batch_idx):
        g = batch
        batchsize = g.num_nodes('global')
        x = self.forward(g)

        targets = [g.nodes['global'].data[k] for k in self.cfg.target_keys]
        target = torch.cat(targets, dim=1)

        loss = self.loss_fn(x, target)

        loss_denom = torch.sum(~torch.isnan(target))
        
        self.log('train_loss', loss, batch_size=loss_denom, on_step=False, on_epoch=True)
        with torch.no_grad():
            mae, mse = self.l1_l2_loss(x, target)
            self.log('train_mae', mae, on_step=False, on_epoch=True, batch_size=loss_denom)
            rmse = torch.sqrt(mse)
            self.log('train_rmse', rmse, on_step=False, on_epoch=True, batch_size=loss_denom)
        return loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        g = batch
        batchsize = g.num_nodes('global')
        preds = self.forward(g)

        targets = [g.nodes['global'].data[k] for k in self.cfg.target_keys]
        predictions = [preds[:, i] for i in range(preds.shape[1])]
        
        # store the targets
        for k, target, pred in zip(self.cfg.target_keys, targets, predictions):
            k_str = f'{k}-{dataloader_idx}' if dataloader_idx > 0 else k
            if k_str not in self.targets:
                self.targets[k_str] = []
                self.predictions[k_str] = []
            self.targets[k_str].append(target)
            self.predictions[k_str].append(pred)

        target = torch.cat(targets, dim=1)

        loss = self.loss_fn(preds, target)

        loss_denom = torch.sum(~torch.isnan(target))

        self.log('val_loss' if dataloader_idx == 0 else f'val_loss-{dataloader_idx}', loss, batch_size=loss_denom, on_step=False, on_epoch=True, add_dataloader_idx=False)

    def on_validation_epoch_start(self):
        self.targets = {}
        self.predictions = {}

    def on_validation_epoch_end(self):
        # calculate mae and rmse for each target key and also for the sum of all target keys
        # further, calculate the pearson correlation coefficient for each target key
        for k in list(self.targets.keys()) + ['all']:
            if k == 'all':
                targets_ = torch.cat([torch.cat(self.targets[k], dim=0) for k in self.cfg.target_keys], dim=0).flatten()
                preds_ = torch.cat([torch.cat(self.predictions[k], dim=0) for k in self.cfg.target_keys], dim=0).flatten()
            else:
                targets_ = torch.cat(self.targets[k], dim=0).flatten()
                preds_ = torch.cat(self.predictions[k], dim=0).flatten()

            mask = ~torch.isnan(targets_)
            if mask.sum() == 0:
                continue
            
            targets = targets_[mask]
            preds = preds_[mask]

            mae = torch.nn.functional.l1_loss(preds, targets)
            rmse = torch.sqrt(torch.nn.functional.mse_loss(preds, targets))
            r = torch.corrcoef(torch.stack([preds, targets]))[0, 1]

            self.log(f'val-{k}/mae', mae, on_step=False, on_epoch=True)
            self.log(f'val-{k}/rmse', rmse, on_step=False, on_epoch=True)
            self.log(f'val-{k}/r', r, on_step=False, on_epoch=True)

            
            # Add scatter plot to wandb using the wandb.plot.scatter method
            # if wandb is used:
            if wandb.run is not None and self.cfg.scatter_plots:
                add_scatter_plot_wand(targets, preds, 'target', 'prediction', f'{k}', f'val-{k}/scatter_plot')
                


    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
    


    def test_step(self, batch, batch_idx, dataloader_idx=0):
        g = batch
        batchsize = g.num_nodes('global')
        print(batchsize)
        preds = self.forward(g)

        targets = [g.nodes['global'].data[k] for k in self.cfg.target_keys]
        predictions = [preds[:, i] for i in range(preds.shape[1])]

        # store the targets
        for k, target, pred in zip(self.cfg.target_keys, targets, predictions):
            k_str = f'{k}-{dataloader_idx}' if dataloader_idx > 0 else k
            if k_str not in self.targets:
                self.targets[k_str] = []
                self.predictions[k_str] = []
            self.targets[k_str].append(target)
            self.predictions[k_str].append(pred)

        target = torch.cat(targets, dim=1)
        loss = self.loss_fn(preds, target)
        loss_denom = torch.sum(~torch.isnan(target))

        if not self.supress_log:
            self.log('test_loss' if dataloader_idx == 0 else f'test_loss-{dataloader_idx}', loss, batch_size=loss_denom, on_step=False, on_epoch=True, add_dataloader_idx=False)

    def on_test_epoch_start(self):
        self.targets = {}
        self.predictions = {}

    def on_test_epoch_end(self):
        metrics_summary = {}

        for k in list(self.targets.keys()) + ['all']:
            if k == 'all':
                targets_ = torch.cat([torch.cat(self.targets[k], dim=0) for k in self.cfg.target_keys], dim=0).flatten()
                preds_ = torch.cat([torch.cat(self.predictions[k], dim=0) for k in self.cfg.target_keys], dim=0).flatten()
            else:
                targets_ = torch.cat(self.targets[k], dim=0).flatten()
                preds_ = torch.cat(self.predictions[k], dim=0).flatten()

            mask = ~torch.isnan(targets_)
            if mask.sum() == 0:
                continue

            targets = targets_[mask]
            preds = preds_[mask]

            mae = torch.nn.functional.l1_loss(preds, targets)
            rmse = torch.sqrt(torch.nn.functional.mse_loss(preds, targets))
            r = torch.corrcoef(torch.stack([preds, targets]))[0, 1]
            r2 = r**2
            std = torch.std(targets)

            # Store metrics in the summary
            metrics_summary[f'{k}/mae'] = mae.item()
            metrics_summary[f'{k}/rmse'] = rmse.item()
            metrics_summary[f'{k}/r'] = r.item()
            metrics_summary[f'{k}/r2'] = r2.item()
            metrics_summary[f'{k}/num_values'] = len(targets.flatten())
            metrics_summary[f'{k}/std'] = std.item()

            targets = targets.cpu().numpy()
            preds = preds.cpu().numpy()

            # Save scatter plot
            save_scatter_plot(targets, preds, 'target', 'prediction', f'{k}', f'{self.test_dir}/test-{k}-scatter.png')

        # add target or prediction to key:
        self.targets = {f'target-{k}': torch.cat(v, dim=0).flatten().cpu().numpy() for k, v in self.targets.items()}
        self.predictions = {f'prediction-{k}': torch.cat(v, dim=0).flatten().cpu().numpy() for k, v in self.predictions.items()}

        # Save metrics and data locally
        np.savez(f'{self.test_dir}/test_results.npz', **self.targets, **self.predictions)
        with open(f'{self.test_dir}/test_metrics_summary.json', 'w') as f:
            json.dump(metrics_summary, f, indent=4)


def save_scatter_plot(x, y, xlabel, ylabel, title, filename):
    # Convert tensors to numpy arrays
    if torch.is_tensor(x):
        x = x.cpu().numpy()
    if torch.is_tensor(y):
        y = y.cpu().numpy()

    global_min = min(x.min(), y.min())
    global_max = max(x.max(), y.max())

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, alpha=0.7)
    plt.plot([global_min, global_max], [global_min, global_max], color='black', linestyle='--', linewidth=1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.title(title)
    plt.grid(True)

    # Save the plot to a file
    plt.savefig(filename)
    plt.close()
    


def add_scatter_plot_wand(x, y, xlabel, ylabel, title, wandb_id):

    # Convert tensors to numpy arrays:
    if torch.is_tensor(x):
        x = x.cpu().numpy()
    if torch.is_tensor(y):
        y = y.cpu().numpy()

    global_min = min(x.min(), y.min())
    global_max = max(x.max(), y.max())

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, alpha=0.7)
    plt.plot([global_min, global_max], [global_min, global_max], color='black', linestyle='--', linewidth=1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)

    # Save plot as an image in-memory and log it to wandb
    wandb.log({wandb_id: wandb.Image(plt)})
    
    # Close the plot to free memory
    plt.close()



