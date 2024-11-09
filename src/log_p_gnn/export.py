#%%
from log_p_gnn.data_utils import dgl_from_cgsmiles, get_hierarchical_graph
from log_p_gnn.pl_module import _get_model_class
from pathlib import Path
import torch
from typing import List, Dict
from omegaconf import OmegaConf
from networkx import Graph


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

        Args:

            cg_smiles: SMILES string representing the CG structure

        Returns:
            Dict[str, float]: Dictionary of logP values
        """
        return _predict(cg_smiles, self.model, self.out_feats)

    def predict_from_nx(self, aa_graph:Graph, cg_graph:Graph)->Dict[str,float]:
        """
        Predict logP values for given AA and CG graphs with the loaded model.
        Returns a dictionary of shape {logP-type: logP-value}

        Args:
            aa_graph: NetworkX graph representing the AA structure
            cg_graph: NetworkX graph representing the CG

        Returns:
            Dict[str, float]: Dictionary of logP values
        """
        return _predict_from_nx(aa_graph, cg_graph, self.model, self.out_feats)

    def __call__(self, *args, **kwds):
        """
        Predict logP values for either a CG SMILES string or a pair of NetworkX graphs representing the AA and CG structures.
        """
        # try to figure out whether the input is a single cgsmiles string or two nx graphs in the order aa,cg, or by keywords:
        # case 1: no keywords, single argument
        if len(args) == 1 and len(kwds) == 0:
            cg_smiles = args[0]
            return self.predict(cg_smiles)
        
        # case 2: no keywords, two arguments
        if len(args) == 2 and len(kwds) == 0:
            aa_graph, cg_graph = args
            return self.predict_from_nx(aa_graph, cg_graph)
        
        # case 3: keywords
        if 'cg_smiles' in kwds:
            assert set(kwds.keys()) == {'cg_smiles'}, f"Expected only 'cg_smiles' as keyword argument, got {kwds.keys()}"
            return self.predict(kwds['cg_smiles'])
        
        if 'aa_graph' in kwds and 'cg_graph' in kwds:
            assert set(kwds.keys()) == {'aa_graph', 'cg_graph'}, f"Expected only 'aa_graph' and 'cg_graph' as keyword arguments, got {kwds.keys()}"
            return self.predict_from_nx(kwds['aa_graph'], kwds['cg_graph'])
        
        raise ValueError("Invalid input arguments. Expected either a single CG SMILES string, or two NetworkX graphs representing the AA and CG structures.")



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

def _predict_from_nx(aa_graph:Graph, cg_graph:Graph, model, out_feats: List[str])->Dict[str,float]:
    g = get_hierarchical_graph(aa_graph, cg_graph, featurize=True)
    with torch.no_grad():
        x = model(g)
    return {k: x[0, i].item() for i, k in enumerate(out_feats)}

