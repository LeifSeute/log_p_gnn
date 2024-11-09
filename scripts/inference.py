#%%
from log_p_gnn.export import LogPGNN
from pathlib import Path
#%%

thisdir = Path(__file__).parent

CKPT_PATH = thisdir.parent / 'exported_ckpts/11-09-2024/best.ckpt'

# instantiate a wrapper class:
logp_gnn = LogPGNN(ckpt_path=CKPT_PATH)


# %%

# predict logP values for a given CG SMILES string:
CG_SMILES = '{[#ALA][#AAMD][#ALA][#ALA][#AAMD][#ALA][#ALA][#2VP]}.{#AAC=[>][#TC3][<][#TP2a],#ALA=[<][#TC3][>][#TN6d],#AAMD=[<][#TC3][>][#TP6a],#MAC=[<][#TC3][>][#SN4a],#STYR=[<][#TC3][>][#TC5]1[#TC5][#TC5]1,#4VP=[<][#TC3][>][#TC5]1[#TN6a][#TC5]1,#2VP=[<][#TC3][>][#TN6aA]1[#TC5][#TC5]1,#12BD=[<][#TC3][>][#SC4]}.{#TC3=[>]CC[$][<],#TP2a=[$]C(=O)OH,#TP6a=[$]C(=O)N,#TN6d=[$]CN,#SC4=[$]C=C,#TC5=[>r]cc[<r][$],#TN6a=[<r]cn[>r],#TN6aA=[<r][$]cn[>r],#SN4a=[$]C(=O)OC}'

logps = logp_gnn.predict(CG_SMILES)
print(logps)
# %%