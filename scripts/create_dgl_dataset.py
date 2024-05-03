if __name__ == '__main__':


    import argparse
    import dgl
    import networkx as nx
    import pandas as pd
    import pysmiles
    import cgsmiles
    from tqdm import tqdm
    from pathlib import Path
    from log_p_gnn.data_utils import homgraph_to_hetgraph, networkx_to_dgl, MAX_BOND_ORDER, ELEMENTS

    thisdir = Path(__file__).parent

    parser = argparse.ArgumentParser(description='Create DGL dataset from excel files.')

    parser.add_argument('--log_p_data', type=str, default=f'{thisdir}/../data/Final MFLOGP Dataset.xlsx',
                        help='Path to the log P data file.')
    parser.add_argument('--cg_data', type=str, default=f'{thisdir}/../data/MartiniCGSmilesDB.xlsx',
                        help='Path to the CG data file.')
    
    parser.add_argument('--output', type=str, default=f'{thisdir}/../data/dgl_dataset.bin',
                        help='Path to the output file.')
    
    args = parser.parse_args()


    def get_mols(log_p_data:str='Final MFLOGP Dataset.xlsx', cg_data:str='MartiniCGSmilesDB.xlsx'):

        ref_df = pd.read_excel(log_p_data, sheet_name='Sheet1')

        LogPs = []

        for mol_name, smiles_str, LogP in zip(ref_df.get('Names'), ref_df.get('SMILES'), ref_df.get('Exp logp')):
            g = pysmiles.read_smiles(smiles_str, explicit_hydrogen=True)
            LogPs.append((g, LogP))

        aa_mols, cg_mols = [], []
        m3_df = pd.read_excel(cg_data, sheet_name='All')
        for mol_name, smiles_str, cgsmiles_str in tqdm(zip(m3_df.get('Name'), m3_df.get('SMILES'), m3_df.get('CGSmiles'))):
            try:
                cg_mol, aa_mol = cgsmiles.resolve.MoleculeResolver(cgsmiles_str).resolve()
            except Exception:
                print(f"Error encountered when reading {mol_name}")
                continue

            try:

                ref_mol = pysmiles.read_smiles(smiles_str, explicit_hydrogen=True)

                def _node_match(node1, node2):
                    return node1['element'] == node2['element']

                if not nx.is_isomorphic(aa_mol, ref_mol, node_match=_node_match):
                    print(f"{mol_name} does not match SMILES.")
                    print(aa_mol.nodes(data='element'))
                    print(ref_mol.nodes(data='element'))
                    break

                for g, LogP in LogPs:
                    if nx.is_isomorphic(g, aa_mol):
                        nx.set_node_attributes(aa_mol, LogP, "logp")
                        nx.set_node_attributes(cg_mol, LogP, "logp")
                        break
                else:
                    raise RuntimeError(f"No match found for {mol_name}")
                
                aa_mols.append(aa_mol)
                cg_mols.append(cg_mol)
            
            except Exception as e:
                print(f"Error encountered when processing {mol_name}: {e}")
                continue

        print(f"Found log P data for {len(aa_mols)} out of {len(m3_df)} molecules.")

        return aa_mols, cg_mols


    if not Path(args.output).parent.exists():
        Path(args.output).parent.mkdir(parents=True)
    
    aa_mols, cg_mols = get_mols(args.log_p_data, args.cg_data)

    graphs = [homgraph_to_hetgraph(networkx_to_dgl(aa_mol), global_feats=['logp']) for aa_mol in tqdm(aa_mols)]

    dgl.save_graphs(args.output, graphs)

    print(f"Saved {len(graphs)} graphs to {Path(args.output).resolve()}")
