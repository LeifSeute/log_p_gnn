#%%
from log_p_gnn.data_utils import get_data_path
from log_p_gnn.training import train_model

config = {
    'gnn':{
        'feats': 32,
        'in_feat_name': [
            'cg_encoding',
            'degree',
            ],     
        'n_att': 2,
        'n_conv': 0,
        'n_heads': 4,
        'attention_dropout': 0.,
        'final_dropout': 0.,
        'initial_dropout': 0.,
        'out_feats': 8,
    },
    'pooling': {
        'max': False,
        'mean': False,
        'sum': True
    },
    'readout': {
        'num_layers': 1,
        'dropout': 0.6,
        'feats': 16
    },
    'denormalizer': {
        'learnable': False
    },
    'training': {
        'lr': 2e-5,
        'noise_level': 0.7, # we add noise to the training data to reduce overfitting
        'dspath': f'{get_data_path()}/cg_dgl_dataset.bin',
        'batch_size': 16,
        'epoch_cycles': 100,
        'name': None,
        'project': 'log_p_gnn-cg',
    },
}

train_model(config)