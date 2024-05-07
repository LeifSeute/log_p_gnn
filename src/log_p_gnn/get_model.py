from log_p_gnn.grappa_model import GrappaGNN
from log_p_gnn.readout import Readout, Denormalize
from log_p_gnn.dgl_pooling import DGLPooling
import torch


def get_model(config, train_target_mean=0, train_target_std=1):

    gnn = GrappaGNN(out_feats=config['gnn']['out_feats'], in_feat_name=config['gnn']['in_feat_name'], in_feat_dims=config['gnn']['in_feat_dims'], n_att=config['gnn']['n_att'], n_heads=config['gnn']['n_heads'], attention_dropout=config['gnn']['attention_dropout'], final_dropout=config['gnn']['final_dropout'], charge_encoding=False, n_conv=config['gnn']['n_conv'], initial_dropout=config['gnn']['initial_dropout'], layer_norm=True)

    pooler = DGLPooling(**config['pooling'])

    pooling_out_dim = sum([config['gnn']['out_feats'] for _ in config['pooling'].values() if _])

    readout = Readout(in_features=pooling_out_dim, out_features=1, hidden_features=config['readout']['feats'], num_layers=config['readout']['num_layers'], dropout=config['readout']['dropout'])

    denormalizer = Denormalize(mean=train_target_mean, std=train_target_std, learnable=config['denormalizer']['learnable'])

    model = torch.nn.Sequential(gnn, pooler, readout, denormalizer)

    return model