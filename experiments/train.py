import hydra
import pytorch_lightning as pl
import torch
import wandb


import os
import torch

import hydra
from omegaconf import DictConfig, OmegaConf

# Pytorch lightning imports
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

import wandb

import utils as eu
from log_p_gnn.test_model import AllAtomGCN
from log_p_gnn.dataset import DataModule
from log_p_gnn.data_utils import get_in_feat_size
from log_p_gnn.pl_module import PLModule
from log_p_gnn.pl_module import _get_model_class

import logging
from pathlib import Path


torch.set_float32_matmul_precision('high')
log = logging.getLogger(__name__)


class Experiment:

    def __init__(self, *, cfg: DictConfig):
        self._cfg = cfg
        self._data_cfg = cfg.data
        self._exp_cfg = cfg.experiment
        self._model_cfg = cfg.model

        OmegaConf.set_struct(self._exp_cfg, True)

        self.create_data_module()
        self.create_module()

    def create_module(self):
        # Load the model class from the config
        model_class = _get_model_class(self._model_cfg.class_path)
        # Instantiate the model using the loaded class
        model: torch.nn.Module = model_class(out_feats=len(self._exp_cfg.training.target_keys), **self._model_cfg.model_args)
        self._model: PLModule = PLModule(model, self._cfg)

    def create_data_module(self):
        self._datamodule: LightningDataModule = DataModule(self._cfg)
        self._datamodule.setup('fit')
        
    def train(self):
        callbacks = []
        if self._exp_cfg.use_wandb:
            logger = WandbLogger(
                **self._exp_cfg.wandb,
            )
        else:
            logger = None
        
        # Checkpoint directory.
        ckpt_dir = self._exp_cfg.checkpointer.dirpath
        os.makedirs(ckpt_dir, exist_ok=True)
        log.info(f"Checkpoints saved to {ckpt_dir}")
        
        # Model checkpoints
        callbacks.append(ModelCheckpoint(**self._exp_cfg.checkpointer))


        # Save config
        cfg_path = os.path.join(ckpt_dir, 'config.yaml')
        with open(cfg_path, 'w') as f:
            OmegaConf.save(config=self._cfg, f=f.name)
        cfg_dict = OmegaConf.to_container(self._cfg, resolve=True)
        flat_cfg = dict(eu.flatten_dict(cfg_dict))
        if self._exp_cfg.use_wandb and isinstance(logger.experiment.config, wandb.sdk.wandb_config.Config):
            logger.experiment.config.update(flat_cfg)

        trainer = Trainer(
            **self._exp_cfg.trainer,
            callbacks=callbacks,
            logger=logger,
            enable_progress_bar=self._exp_cfg.use_tqdm,
            enable_model_summary=True,
        )
        trainer.fit(
            model=self._model,
            datamodule=self._datamodule,
            ckpt_path=self._exp_cfg.warm_start
        )

    def test(self, ckpt_path: str, eval_all=False):
        self.trainer = Trainer(
            **self._exp_cfg.trainer,
            logger=False,
            enable_progress_bar=self._exp_cfg.use_tqdm,
            enable_model_summary=False,
            devices=1,
            num_nodes=1,
            strategy='auto',
        )

        self._datamodule.setup('test')
        
        # test the model
        self._model.supress_log = True
        self._model.test_dir = Path(ckpt_path).parent/'test_set'
        self._model.test_dir.mkdir(exist_ok=True, parents=True)
        self.trainer.test(
            model=self._model,
            dataloaders=self._datamodule.test_dataloader(),
            ckpt_path=ckpt_path
        )

        if not eval_all:
            return
        
        # now do the same for train and val (a bit hacky but lightning doesnt offer a simpler solution):

        self._model.test_dir = Path(ckpt_path).parent/'train_set'
        self._model.test_dir.mkdir(exist_ok=True, parents=True)
        self.trainer.test(
            model=self._model,
            dataloaders=self._datamodule.train_dataloader(),
            ckpt_path=ckpt_path
        )

        # n = 0
        # for batch in self._datamodule.train_dataloader():
        #     n += batch.num_nodes('global')

        # print(n)
        # return

        self._model.test_dir = Path(ckpt_path).parent/'val_set'
        self._model.test_dir.mkdir(exist_ok=True, parents=True)
        self.trainer.test(
            model=self._model,
            dataloaders=self._datamodule.val_dataloader(),
            ckpt_path=ckpt_path
        )


@hydra.main(version_base=None, config_path="../configs", config_name="train_hgcn.yaml")
def main(cfg: DictConfig):
    
    if cfg.experiment.warm_start is not None and cfg.experiment.warm_start_cfg_override:
        # Loads warm start config.
        warm_start_cfg_path = os.path.join(
            os.path.dirname(cfg.experiment.warm_start), 'config.yaml')
        warm_start_cfg = OmegaConf.load(warm_start_cfg_path)

        # Warm start config may not have latest fields in the base config.
        # Add these fields to the warm start config.
        OmegaConf.set_struct(cfg.model, False)
        OmegaConf.set_struct(warm_start_cfg.model, False)
        cfg.model = OmegaConf.merge(cfg.model, warm_start_cfg.model)
        OmegaConf.set_struct(cfg.model, True)
        log.info(f'Loaded warm start config from {warm_start_cfg_path}')

    exp = Experiment(cfg=cfg)
    exp.train()

    exp.test(str(Path(cfg.experiment.checkpointer.dirpath)/'best.ckpt'))

if __name__ == "__main__":
    main()
