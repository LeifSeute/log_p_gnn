#%%
from log_p_gnn.data_utils import get_data_path
from log_p_gnn.training import train_model

config = {
    'gnn':{
        'feats': 64,
        'in_feat_name': ['atomic_number', 'is_aromatic', 'total_charge', 'degree'],     
        'n_att': 2,
        'n_conv': 0,
        'n_heads': 8,
        'attention_dropout': 0.,
        'final_dropout': 0.,
        'initial_dropout': 0.,
        'out_feats': 64,
    },
    'pooling': {
        'max': False,
        'mean': False,
        'sum': True
    },
    'readout': {
        'num_layers': 1,
        'dropout': 0.2,
        'feats': 64
    },
    'denormalizer': {
        'learnable': False
    },
    'training': {
        'lr': 2e-5,
        'noise_level': 0.2,
        'dspath': f'{get_data_path()}/all_atom_dgl_dataset.bin',
        'batch_size': 32,
        'epoch_cycles': 10,
        'name': None,
        'project': 'log_p_gnn-all-atom',
    },
}

train_model(config)