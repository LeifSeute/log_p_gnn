#%%
import networkx as nx
from cgsmiles import read_cgsmiles
import itertools
import numpy as np
from collections import Counter, defaultdict
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import pysmiles
import cgsmiles
import optuna
import matplotlib.pyplot as plt
from pathlib import Path

ckpt_dir = '../exported_ckpts/extended-12-11-2024'

df_dir = '../data/all_logP_v5_extended.csv'

this_dir = Path(__file__).parent

#%%
def read_inter_matrix(datafile="levels.dat"):
    with open(datafile, 'r') as _file:
        lines = _file.readlines()
        level_dict = {}
        col_labels = lines[0].strip().split() 
        bead_to_index = dict(zip(col_labels, np.arange(0, len(col_labels))))
        for line in lines[1:]:
            tokens = line.strip().split()
            row_type = tokens[-1]
            for idx, token in enumerate(tokens[:-1]):
                level_dict[frozenset([row_type, col_labels[idx]])] = int(token)
    return level_dict, bead_to_index

def read_mol_strings(datafile='mols.dat'):
    mol_strings = {}
    with open(datafile, 'r') as _file:
        lines = _file.readlines()
    for line in lines:
        tokens = line.strip().split()
        mol_strings[tokens[0]] = tokens[1]
    return mol_strings

def modify_level(level, ba, bb, alA, alB, dlA, dlB, rlA, rlB, hlA, hlB):
    
    if ba == 'W' and (alB or dlB):
        level += 1
        
    if bb == 'W' and (alA or dlA):
        level += 1
    
    if alA and alB:
        level += 1
            
    if dlA and dlB:
        level += 1
        
    if (alA and dlB) or (dlA and alB):
        level -= 1
            
    if rlA and rlB:
        level += 1
            
    if hlA and hlB:
        level -= 1
            
    return level


def generate_feature_vector(mol_str_a, mol_str_b, bead_matrix, bead_to_index, polyA=False, polyB=False):
    """
    Generate the graph of a Martini molecule and
    assing bead types.
    """
    feat_level_matrix_W = np.zeros((19, 6),dtype=float)
    feat_level_matrix_SOL = np.zeros((19, 6),dtype=float)
    feat_bead_count_vector = np.zeros((1), dtype=float)
    size_to_plane = {frozenset((0, 0)): 0,
                     frozenset((1, 1)): 1,
                     frozenset((2, 2)): 2,
                     frozenset((0, 1)): 3,
                     frozenset((0, 2)): 4,
                     frozenset((1, 2)): 5,}

    # print(mol_str_a)
    #mol_a = read_cgsmiles(mol_str_a)
    try:
        mol_a, aa_mol = cgsmiles.resolve.MoleculeResolver.from_string(mol_str_a, legacy=True).resolve_all()
    except:
        print(mol_str_a)
        raise
    smiles_str = pysmiles.write_smiles(aa_mol)
   # print(smiles_str)
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles_str))
    AllChem.EmbedMolecule(mol)
    vol = AllChem.ComputeMolVolume(mol)
    
    mol_b = read_cgsmiles(mol_str_b)
   # print(mol_str_a, mol_str_b)
    mol_a_types = list(nx.get_node_attributes(mol_a, "fragname").values())
    mol_b_types = list(nx.get_node_attributes(mol_b, "fragname").values())
    feat_bead_count_vector = vol #len(mol_a_types)
    #print(mol_a_types)
   # print(mol_b_types)
    inter_graphs = []
    for mol in [mol_a_types, mol_b_types, ['W']]:
       # print(mol)
        inter_graph = nx.Graph()
        idx = 0
        #bead_counts = Counter(mol)
        #Efor bead, num in bead_counts.items():
        for bead in mol:
            if bead[0] == "T":
                size = 0
                bead = bead[1:]
            elif bead[0] == "S":
                size = 1
                bead = bead[1:]
            else:
                size = 2 
                        
            al = False
            dl = False
            hl = False
            rl = False
            el = False
            vl = False
            
            if bead[-1].isupper() and bead[-1] != 'W':
                bead = bead[:-1]
            
            if bead[-1] == "r":
                rl = True
                bead = bead[:-1]
            
            if bead[-1] == "h":
                hl = True
                bead = bead[:-1]
            
            if bead[-1] == 'e':
                el = True
                bead = bead[:-1]
            
            if bead[-1] == 'v':
                vl = True
                bead = bead[:-1]
        
            if bead[-1] == "a":
                al = True
                bead = bead[:-1]

            if bead[-1] == "d":
                dl = True
                bead = bead[:-1]

        
            bead_index = bead_to_index[bead]        
            inter_graph.add_node(idx, size=size, bead_type=bead, alabel=al, dlabel=dl, rlabel=rl, hlabel=hl)
            idx += 1
        inter_graphs.append(inter_graph)
    #print(len(inter_graphs))
   # print(inter_graphs[0].nodes(data=True))
        
    # combinations of W with solute
    # combinations of solvent and solute
    bead_combos = []
    counter = 0
    for inter_graph_a, inter_graph_b in [(inter_graphs[0],inter_graphs[1]), (inter_graphs[0], inter_graphs[2])]:
       # print(inter_graph_a.nodes(data=True))
       # print(inter_graph_b.nodes(data=True))
       # print("\n")
        bead_combos = itertools.product(inter_graph_a.nodes, inter_graph_b.nodes)
        #print(list(bead_combos))
        bead_combos = itertools.product(inter_graph_a.nodes, inter_graph_b.nodes)
        for ba, bb in bead_combos:
            bead_a = inter_graph_a.nodes[ba]['bead_type']
            bead_b = inter_graph_b.nodes[bb]['bead_type']
            level = bead_matrix[frozenset([bead_a, bead_b])]
            #if level in [3,4,5,6,7,8]:
            #    print(bead_a, bead_b, level)
            level = modify_level(level=level,
                                 ba=bead_a,
                                 bb=bead_b,
                                 alA=inter_graph_a.nodes[ba]['alabel'], 
                                 alB=inter_graph_b.nodes[bb]['alabel'], 
                                 dlA=inter_graph_a.nodes[ba]['dlabel'], 
                                 dlB=inter_graph_b.nodes[bb]['dlabel'], 
                                 rlA=inter_graph_a.nodes[ba]['rlabel'], 
                                 rlB=inter_graph_b.nodes[bb]['rlabel'], 
                                 hlA=inter_graph_a.nodes[ba]['hlabel'], 
                                 hlB=inter_graph_b.nodes[bb]['hlabel'],)
            #print(bead_a, bead_b, level)
            third_dimension = size_to_plane[frozenset([inter_graph_a.nodes[ba]['size'], 
                                                       inter_graph_b.nodes[bb]['size']])]
        
            if counter == 0:
                feat_level_matrix_SOL[level-1, third_dimension] += 1
            else:
                feat_level_matrix_W[level-1, third_dimension] += 1

        counter += 1
       # if polyA:
       #     feat_bead_count_vector[-2] = 1
       # if polyB:
       #     feat_bead_count_vector[-1] = 1
            
        
            
    #feat_level_matrix = np.sum(feat_level_matrix, axis=2)
   # print(feat_level_matrix.shape)
    #print(feat_level_matrix.shape)
    feature_vector = feat_level_matrix_SOL.reshape(-1)
    #print(feature_vector.shape)
    feature_vector = np.hstack([feat_level_matrix_SOL.reshape(-1), 
                                feat_level_matrix_W.reshape(-1)])
    #print(feature_vector.shape)
    #print(feat_bead_count_vector.shape)
    feature_vector = np.hstack([feature_vector, np.array([feat_bead_count_vector])])
    
    return feature_vector, bead_to_index


#%%
m3_df = pd.read_csv(df_dir)

import json

with open(ckpt_dir + '/train_mol_tags.json', 'r') as f:
    train_mols = json.load(f)

with open(ckpt_dir + '/test_mol_tags.json', 'r') as f:
    test_mols = json.load(f)

with open(ckpt_dir + '/val_mol_tags.json', 'r') as f:
    val_mols = json.load(f)

#%%

train_indices = m3_df.index[m3_df["cgsmiles_str"].isin(train_mols)]
test_indices = m3_df.index[m3_df["cgsmiles_str"].isin(test_mols)]
val_indices = m3_df.index[m3_df["cgsmiles_str"].isin(val_mols)]

print('train mols:', len(train_indices))
print('test mols:', len(test_indices))
print('val mols:', len(val_indices))
print('total mols:', len(train_indices) + len(test_indices) + len(val_indices), ' vs ', len(m3_df))

df_train = m3_df.loc[train_indices]
df_test = m3_df.loc[test_indices]
df_val = m3_df.loc[val_indices]
mols_train = df_train['cgsmiles_str']
mols_test = df_test['cgsmiles_str']
mols_val = df_val['cgsmiles_str']


# calculate number of data points per solvent:
train_oco = df_train['OCO'].values
test_oco = df_test['OCO'].values
val_oco = df_val['OCO'].values

print('train oco total:', len(train_oco))

# remove nans:
nan_train_oco = np.isnan(train_oco)
train_oco = train_oco[~np.isnan(train_oco)]
test_oco = test_oco[~np.isnan(test_oco)]
val_oco = val_oco[~np.isnan(val_oco)]

print('train oco without nan:', len(train_oco))
print('test oco without nan:', len(test_oco))
print('val oco without nan:', len(val_oco))

#%%

from tqdm import tqdm


def generate_fingerprint_vectors(df, mol_labels):
    level_matrix, bead_to_index = read_inter_matrix('./levels.dat')
    # mol_strings = read_mol_strings() 

    n_oco = df["OCO"].notna().sum()
    n_hd =  df["HD"].notna().sum()
    n_clf =  df["CLF"].notna().sum()
    print('n_oco', n_oco)
    print('n_hd', n_hd)
    print('n_clf', n_clf)
    
    # fingerprint vectors
    vector_OCO = np.zeros((n_oco, 229), dtype=int)
    vector_HD = np.zeros((n_hd, 229), dtype=int)
    vector_CLF = np.zeros((n_clf, 229), dtype=int)

    # labels
    y_OCO = np.zeros(n_oco, dtype=float)
    y_HD = np.zeros(n_hd, dtype=float)
    y_CLF = np.zeros(n_clf, dtype=float)

    idx_oco = 0
    idx_hd = 0
    idx_clf = 0

    mols = []
    feature_list = []
    mol_counter = 0
    diff_counter = 0

    # CGSmiles of Solvent Molecules
    oco="{[#SC2][#SC2][#SP2]}"
    hd="{[#C1]|4}"
    clf="{[#X2]}"
    
    for mol_tag, cgsmiles_str, OCO, HD, CLF in tqdm(list(zip(mol_labels, 
        df.get('cgsmiles_str'), 
        df.get('OCO'),  
        df.get('HD'),  
        df.get('CLF')))):
  
        cgmol = cgsmiles_str 
        for soltyp, mol_str_b, logp in zip(['oco', 'clf', 'hd'], [oco, clf, hd], [OCO, CLF, HD]):
            logp = float(logp)
            
            if np.isnan(logp):
                continue
            
            try:
                new_vector, bead_to_index = generate_feature_vector(cgmol, 
                    mol_str_b, 
                    level_matrix, 
                    bead_to_index, 
                    polyB=False)
            except Exception as e:
                print('Error:', e)
                print(mol_tag)
                raise e
                    
            if soltyp == 'oco':
                vector_OCO[idx_oco, :] = new_vector[:]
                y_OCO[idx_oco] = float(logp)
                idx_oco +=1     
            elif soltyp == 'clf':
                vector_CLF[idx_clf, :] = new_vector[:]
                y_CLF[idx_clf] = float(logp)
                idx_clf +=1                  
            else:
                vector_HD[idx_hd, :] = new_vector[:]
                y_HD[idx_hd] = float(logp)
                idx_hd +=1  

        mols.append((mol_tag, cgsmiles_str, mol_str_b))

    return vector_OCO, vector_HD, vector_CLF, y_OCO, y_HD, y_CLF


#%%

# LOAD RF DATA
############################################

vector_OCO_train, vector_HD_train, vector_CLF_tain, y_OCO_train, y_HD_train, y_CLF_train = generate_fingerprint_vectors(df_train, mols_train)

vector_OCO_test, vector_HD_test, vector_CLF_test, y_OCO_test, y_HD_test, y_CLF_test = generate_fingerprint_vectors(df_test, mols_test)

vector_OCO_val, vector_HD_val, vector_CLF_val, y_OCO_val, y_HD_val, y_CLF_val = generate_fingerprint_vectors(df_val, mols_val)


v_train_all = np.vstack((vector_OCO_train, vector_HD_train, vector_CLF_tain))
y_train_all = np.hstack((y_OCO_train, y_HD_train, y_CLF_train))

v_test_all = np.vstack((vector_OCO_test, vector_HD_test, vector_CLF_test))
y_test_all = np.hstack((y_OCO_test, y_HD_test, y_CLF_test))

v_val_all = np.vstack((vector_OCO_val, vector_HD_val, vector_CLF_val))
y_val_all = np.hstack((y_OCO_val, y_HD_val, y_CLF_val))

############################################

#%%


# OPTIMIZE RF
############################################


import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score    
from sklearn.metrics import r2_score, mean_absolute_error
import optuna



def objective(trial):
    # Define the hyperparameters to be tuned
    n_estimators = trial.suggest_int('n_estimators', 10, 300)
    max_depth = trial.suggest_int('max_depth', 10, 50)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 50)

    MF = trial.suggest_categorical('max_features', ['sqrt', 'log2'])
    
    
    
    # Initialize the model with the suggested hyperparameters
    model = RandomForestRegressor(n_estimators=n_estimators, 
                                   max_depth=max_depth, 
                                   max_features=MF,
                                   min_samples_split=min_samples_split)

    # Evaluate the model using cross-validation
    score_mea = cross_val_score(model, v_val_all, y_val_all, cv=5, scoring='neg_mean_absolute_error').mean()
    score_r2 = cross_val_score(model, v_val_all, y_val_all, cv=5, scoring='r2').mean()
    score = -score_mea + 10*(1-score_r2)
    score = score_r2
    score = score_mea
    return score

# Create the study object and optimize the objective function
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# Best parameters
settings = study.best_params
print(settings)


clf = RandomForestRegressor(**settings)
clf.fit( v_train_all, y_train_all)


#%%


from sklearn.metrics import r2_score, mean_absolute_error
import optuna
y_pred = clf.predict(v_val_all)

def objective(trial):
    # Define the hyperparameters to be tuned
    a = trial.suggest_float('a', -4, 4)
    b = trial.suggest_float('b', -2, 2)

    def correction(x, a, b):
        return x+a*x+b    
    y_corrected = correction(y_pred, a, b)
    
    score = mean_absolute_error(y_val_all, y_corrected)+5*(1-r2_score(y_val_all, y_corrected))
    #score = 1-r2_score(y_val_all, y_corrected)
    return score

# Create the study object and optimize the objective function
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

# Best parameters
params = study.best_params

############################################

#%%

#params={'a': 1.134997891196554, 'b': -1.6173264505748035}

def correction(x, a, b):
        return x+a*x+b

evaluation = {'MEA':{}, 'R2':{}}

for label, test, y in zip(['OCO', 'CLF', 'HD', 'all'], 
                           [vector_OCO_test, vector_CLF_test, vector_HD_test, v_test_all],
                           [y_OCO_test, y_CLF_test, y_HD_test, y_test_all]):
    mea = mean_absolute_error(y,correction(clf.predict(test), **params)) 
    r2= r2_score(y, correction(clf.predict(test), **params))
    evaluation['MEA'][label]=round(mea,2)
    evaluation['R2'][label]=round(r2,2)



#%%

# PLOT RF
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, gridspec_kw={'wspace': 0}, figsize=(12, 5))

ntrain_OCO=vector_OCO_train.shape[0]
ntest_OCO=vector_OCO_test.shape[0]

ax1.scatter(y_OCO_train, correction(clf.predict(vector_OCO_train),  **params), 
            marker='o', facecolors='none', edgecolors='tab:blue', label=f"train set (n={ntrain_OCO})")
#ax1.scatter(y_OCO_train, clf.predict(vector_OCO_train), 
#            marker='o', facecolors='none', edgecolors='tab:green', label=f"train set (n={ntrain_OCO})")

ax1.scatter(y_OCO_test, correction(clf.predict(vector_OCO_test),  **params), 
            marker='s',facecolors='orange', edgecolors='black', label=f"test set (n={ntest_OCO})")
ax1.text(10, -50, s="MAE: " + str(evaluation['MEA']['OCO']) +"kJ/mol")
ax1.text(10, -60, s="$R^2$: " + str(evaluation['R2']['OCO']))

ax1.set_xlim(-50, 60)

ntrain_CLF=vector_CLF_tain.shape[0]
ntest_CLF=vector_CLF_test.shape[0]
ax2.scatter(y_CLF_train, correction(clf.predict(vector_CLF_tain),  **params), 
            marker='o', facecolors='none', edgecolors='tab:blue', label=f"train set (n={ntrain_CLF})")
ax2.scatter(y_CLF_test, correction(clf.predict(vector_CLF_test),  **params), 
            marker='s', facecolors='orange', edgecolors='black', label=f"test set (n={ntest_CLF})")
ax2.text(10, -50, s="MAE: " + str(evaluation['MEA']['CLF']) +"kJ/mol")
ax2.text(10, -60, s="$R^2$: " + str(evaluation['R2']['CLF']))

ax2.set_xlim(-50, 60)
ntrain_HD=vector_HD_train.shape[0]
ntest_HD=vector_HD_test.shape[0]
ax3.scatter(y_HD_train, correction(clf.predict(vector_HD_train),  **params), 
            marker='o', facecolors='none', edgecolors='tab:blue', label=f"train set (n={ntrain_HD})")
ax3.scatter(y_HD_test, correction(clf.predict(vector_HD_test),  **params), 
            marker='s',facecolors='orange',  edgecolors='black', label=f"test set (n={ntest_HD})")
ax3.text(10, -50, s="MAE: " + str(evaluation['MEA']['HD']) +"kJ/mol")
ax3.text(10, -60, s="$R^2$: " + str(evaluation['R2']['HD']))

ax3.set_xlim(-50, 60)

for ax in [ax1, ax2, ax3]:
    ax.plot([-45 , 55], [-45, 55], c='black', ls='--')

ax1.set(xlabel='$\Delta$G reference (kJ/mol)', ylabel='$\Delta$G predicted (kJ/mol)')
ax2.set(xlabel='$\Delta$G reference (kJ/mol)')
ax3.set(xlabel='$\Delta$G reference (kJ/mol)')
ax1.legend()
ax2.legend()
ax3.legend()
plt.savefig(this_dir/'RF_results.png', dpi=300)
# %%

# PLOT GNN

import numpy as np
gnn_data_test=dict(np.load(f"{ckpt_dir}/test_set/test_results.npz"))
gnn_data_train=dict(np.load(f"{ckpt_dir}/train_set/test_results.npz"))
# %%
from sklearn.metrics import r2_score, mean_absolute_error
evaluation = {'MEA':{}, 'R2':{}}

for label in ['OCO', 'CLF', 'HD']:
    print()
    gnn_data_test['prediction-'+label] = gnn_data_test['prediction-'+label][~np.isnan(gnn_data_test['target-'+label])]
    gnn_data_test['target-'+label] = gnn_data_test['target-'+label][~np.isnan(gnn_data_test['target-'+label])]

    gnn_data_train['prediction-'+label] = gnn_data_train['prediction-'+label][~np.isnan(gnn_data_train['target-'+label])]
    gnn_data_train['target-'+label] = gnn_data_train['target-'+label][~np.isnan(gnn_data_train['target-'+label])]
    
    ypred = gnn_data_test['prediction-'+label]
    y = gnn_data_test['target-'+label]



    
    mea = mean_absolute_error(y, ypred) 
    r2= r2_score(y, ypred)
    evaluation['MEA'][label]=round(mea,2)
    evaluation['R2'][label]=round(r2,2)

#%%

import matplotlib.pyplot as plt

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, gridspec_kw={'wspace': 0}, figsize=(12, 5))

ntrain_OCO=gnn_data_train['target-OCO'].shape[0]
ntest_OCO=gnn_data_test['target-OCO'].shape[0]

ax1.scatter(gnn_data_train['target-OCO'], gnn_data_train['prediction-OCO'], 
            marker='o', facecolors='none', edgecolors='tab:blue', label=f"train set (n={ntrain_OCO})")
ax1.scatter(gnn_data_test['target-OCO'], gnn_data_test['prediction-OCO'],
            marker='s', facecolors='orange', edgecolors='black', label=f"test set (n={ntest_OCO})")
ax1.text(10, -50, s="MAE: " + str(evaluation['MEA']['OCO']) +"kJ/mol")
ax1.text(10, -60, s="$R^2$: " + str(evaluation['R2']['OCO']))

ax1.set_xlim(-50, 60)

ntrain_CLF=gnn_data_train['target-CLF'].shape[0]
ntest_CLF=gnn_data_test['target-CLF'].shape[0]
ax2.scatter(gnn_data_train['target-CLF'], gnn_data_train['prediction-CLF'], 
            marker='o', facecolors='none', edgecolors='tab:blue', label=f"train set (n={ntrain_CLF})")
ax2.scatter(gnn_data_test['target-CLF'], gnn_data_test['prediction-CLF'],
            marker='s', facecolors='orange', edgecolors='black', label=f"test set (n={ntest_CLF})")
ax2.text(10, -50, s="MAE: " + str(evaluation['MEA']['CLF']) +"kJ/mol")
ax2.text(10, -60, s="$R^2$: " + str(evaluation['R2']['CLF']))

ax2.set_xlim(-50, 60)
ntrain_HD=gnn_data_train['target-HD'].shape[0]
ntest_HD=gnn_data_test['target-HD'].shape[0]
ax3.scatter(gnn_data_train['target-HD'], gnn_data_train['prediction-HD'], 
            marker='o', facecolors='none', edgecolors='tab:blue', label=f"train set (n={ntrain_HD})")
ax3.scatter(gnn_data_test['target-HD'], gnn_data_test['prediction-HD'], 
            marker='s', facecolors='orange', edgecolors='black', label=f"test set (n={ntest_HD})")
ax3.text(10, -50, s="MAE: " + str(evaluation['MEA']['HD']) +"kJ/mol")
ax3.text(10, -60, s="$R^2$: " + str(evaluation['R2']['HD']))

ax3.set_xlim(-50, 60)
ax1.set(xlabel='$\Delta$G reference (kJ/mol)', ylabel='$\Delta$G predicted (kJ/mol)')
ax2.set(xlabel='$\Delta$G reference (kJ/mol)')
ax3.set(xlabel='$\Delta$G reference (kJ/mol)')
ax1.legend()
ax2.legend()
ax3.legend()
for ax in [ax1, ax2, ax3]:
    ax.plot([-45 , 60], [-45, 60], c='black', ls='--')
plt.savefig(this_dir/'GNN_results.png', dpi=300)
# %%
