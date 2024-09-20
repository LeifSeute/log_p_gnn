#%%
from log_p_gnn.data_utils import dgl_from_cgsmiles
from log_p_gnn.pl_module import _get_model_class
from pathlib import Path
import torch
from typing import List, Dict
from omegaconf import OmegaConf


# define a wrapper class:
class LogPGNN:
    def __init__(self, ckpt_path: Path):
        """
        Load a trained model from a checkpoint and prepare it for prediction.
        Expects a config.yaml file in the same directory as the checkpoint.
        """
        self.model = load_model(ckpt_path)
        self.out_feats = get_out_feats(ckpt_path)

    def predict(self, cg_smiles: str)->Dict[str,float]:
        """
        Predict logP values for a given CG SMILES string with the loaded model.
        Returns a dictionary of shape {logP-type: logP-value}
        """
        return _predict(cg_smiles, self.model, self.out_feats)



# %%
def load_model(ckpt_path: Path):
    """
    Load a model from a checkpoint. Expects a config.yaml file in the same directory as the checkpoint.
    """

    # load the config:
    assert ckpt_path.exists(), f"Checkpoint path {ckpt_path} does not exist."
    config_path = ckpt_path.parent / 'config.yaml'
    assert config_path.exists(), f"Config path {config_path} does not exist."

    # Load the config
    cfg = OmegaConf.load(config_path)

    model_cfg = cfg.model
    exp_cfg = cfg.experiment

    # Load the model class from the config
    model_class = _get_model_class(model_cfg.class_path)
    # Instantiate the model using the loaded class
    model: torch.nn.Module = model_class(out_feats=len(exp_cfg.training.target_keys), **model_cfg.model_args)

    # load the weights:
    state_dict = torch.load(ckpt_path)['state_dict']

    # remove the trailing 'model.' from the keys:
    state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model = model.eval()
    return model


def get_out_feats(ckpt_path: Path):
    """
    Get the order of output features from a model checkpoint for translating model output to logP values.
    """
    # load the config:
    assert ckpt_path.exists(), f"Checkpoint path {ckpt_path} does not exist."
    config_path = ckpt_path.parent / 'config.yaml'
    assert config_path.exists(), f"Config path {config_path} does not exist."

    # Load the config
    cfg = OmegaConf.load(config_path)

    return cfg.experiment.training.target_keys


def _predict(cg_smiles:str, model, out_feats: List[str])->Dict[str,float]:
    g = dgl_from_cgsmiles(cg_smiles)
    with torch.no_grad():
        x = model(g)
    return {k: x[0, i].item() for i, k in enumerate(out_feats)}