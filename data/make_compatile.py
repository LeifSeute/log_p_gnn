#%%
import pandas as pd

p = 'large_polymers.ss'

#%%
df = pd.read_csv(p, delim_whitespace=True)

df['mol_tag'] = df['set']
df['mol_name'] = df['set']

df['OCO'] = df['dG']

df['HD'] = [None]*len(df)

df['CLF'] = [None]*len(df)

# sort the columns: mol_tag, mol_name, OCO, HD, CLF

df = df[['mol_tag', 'mol_name', 'cgsmiles_str', 'OCO', 'HD', 'CLF']]


# %%
df.to_csv('large_polymers_corrected.ss', sep=' ', index=False)
# %%
df_new = pd.read_csv('large_polymers_corrected.ss', delim_whitespace=True)
# %%
