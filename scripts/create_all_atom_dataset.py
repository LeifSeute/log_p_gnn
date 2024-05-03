if __name__ == '__main__':
    

    import argparse
    import dgl
    import networkx as nx
    import pandas as pd
    import pysmiles
    from tqdm import tqdm
    from pathlib import Path

    from log_p_gnn.data_utils import homgraph_to_hetgraph, networkx_to_dgl, MAX_BOND_ORDER, ELEMENTS


    thisdir = Path(__file__).parent

    parser = argparse.ArgumentParser(description='Create DGL dataset from excel files.')

    parser.add_argument('--log_p_data', type=str, default=f'{thisdir}/../data/Final MFLOGP Dataset.xlsx',
                        help='Path to the log P data file.')
    
    parser.add_argument('--output', type=str, default=f'{thisdir}/../data/all_atom_dgl_dataset.bin',
                        help='Path to the output file.')
    
    args = parser.parse_args()


    def get_mols(log_p_data:str='Final MFLOGP Dataset.xlsx'):

        ref_df = pd.read_excel(log_p_data, sheet_name='Sheet1')

        mols = []

        for mol_name, smiles_str, LogP in zip(ref_df.get('Names'), ref_df.get('SMILES'), ref_df.get('Exp logp')):
            g = pysmiles.read_smiles(smiles_str, explicit_hydrogen=True)
            nx.set_node_attributes(g, LogP, "logp")

            mols.append(g)

        print(f"Found log P data for {len(mols)} molecules.")

        return mols
    

    if not Path(args.output).parent.exists():
        Path(args.output).parent.mkdir(parents=True)
    
    aa_mols = get_mols(args.log_p_data)

    print('Converting molecules to DGL graphs...')

    graphs = [homgraph_to_hetgraph(networkx_to_dgl(aa_mol), global_feats=['logp']) for aa_mol in tqdm(aa_mols)]

    dgl.save_graphs(args.output, graphs)

    print(f"Saved {len(graphs)} graphs to {Path(args.output).resolve()}")
