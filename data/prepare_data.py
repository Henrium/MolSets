from rdkit import Chem
from pymatgen.core.periodic_table import Element
from torch_geometric.data import Data
import torch
import pandas as pd

def smiles_to_graph(smiles, encoding='other'):
    mol = Chem.MolFromSmiles(smiles)
    num_nodes = mol.GetNumAtoms()
    # Get node features
    node_features = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol in ['Au', 'Cu']:
            atom.SetAtomicNum(6)
            symbol = 'C'
        element = Element(symbol)
        if encoding == 'onehot':
            node_feature = [int(symbol == x) for x in ['C', 'N', 'O', 'S', 'F', 'Cl']]  # one-hot   encoding of atom type
            node_feature += [atom.GetTotalNumHs(), atom.GetFormalCharge(), atom.GetMass()]  # add some descriptors
            node_features.append(node_feature)
        elif encoding == 'atomic':
            node_feature = [atom.GetAtomicNum(), atom.GetMass(), atom.GetFormalCharge(), atom.GetTotalNumHs()]
            node_feature += [element.X, element.van_der_waals_radius]
            # Add more?
            node_features.append(node_feature)
        else:
            node_feature = [int(symbol == x) for x in ['C', 'N', 'O', 'B', 'S', 'F', 'Cl']]
            # These contain redundant info
            node_feature += [atom.GetAtomicNum(), atom.GetMass(), atom.GetFormalCharge(), atom.GetTotalNumHs()]
            node_feature += [element.X, element.van_der_waals_radius]
            node_features.append(node_feature)
    node_features = torch.tensor(node_features, dtype=torch.float)
    
    # Get edge features
    edge_index = []
    edge_weight = []
    for bond in mol.GetBonds():
        start_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        edge_index.append([start_idx, end_idx])
        edge_index.append([end_idx, start_idx])
        edge_weight.append(bond.GetBondTypeAsDouble())
        edge_weight.append(bond.GetBondTypeAsDouble())

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)

    # Add self-loops manually
    edge_index_loop = torch.tensor([[i, i] for i in range(num_nodes)], dtype=torch.long).t().contiguous()
    edge_weight_loop = torch.ones(num_nodes, dtype=torch.float)
    edge_index = torch.cat([edge_index, edge_index_loop], dim=1)
    edge_weight = torch.cat([edge_weight, edge_weight_loop])

    # Create PyTorch Geometric Data object
    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_weight, num_nodes=num_nodes)

    return data

test = smiles_to_graph('[Cu]OCC[Au]')

def prepare_data(data_file, props):
    data_all = pd.read_csv(data_file, index_col=0)
    index_list = []
    graph_list = []
    target_list = []
    salt_mol = []
    salt_graphs = []
    frac_list = []  # Weight fraction of each polymer
    mw_list = []
    data_all['RT_Cond'] = data_all['Slope'] / 298 + data_all['Intercept']
    means = dict()
    stds = dict()
    means['Slope'], means['Intercept'], means['RT_Cond'] = data_all[['Slope', 'Intercept', 'RT_Cond']].mean()
    stds['Slope'], stds['Intercept'], stds['RT_Cond'] = data_all[['Slope', 'Intercept', 'RT_Cond']].std()
    for i, row in data_all.iterrows():
        mw_list.append(eval(row['M']))
        index_list.append(i)
        smiles = [row[s] for s in ['Solvent 1', 'Solvent 2', 'Solvent 3', 'Solvent 4'] if row[s] == row[s]]
        graphs = list(map(smiles_to_graph, smiles))
        graph_list.append(graphs)
        # target_list.append([(row[prop] - means[prop]) / stds[prop] for prop in props])
        target_list.append([(row[prop] - 0) / 1 for prop in props])
        
        fraction = eval(row['Weights'])
        # frac_list.append([w / sum(fraction) for w in fraction])
        salt_mol.append(row['Molality'])
        salt_graphs.append(smiles_to_graph(row['Salt']))
        # frac_list.append([w / sum(fraction) for w in fraction])
        frac_list.append(fraction)

    data_list = list(zip(index_list, graph_list, mw_list, frac_list, salt_mol, salt_graphs, target_list))
    return data_list

data_list = prepare_data('data_compiled.csv', props=['RT_Cond'])
import pickle
with open('data_list.pkl', 'wb') as f:
    pickle.dump(data_list, f)


df = pd.DataFrame(data_list, columns=['index', 'graphs', 'mw', 'fraction', 'salt_mol', 'salt', 'target'], dtype=object)
df.set_index('index', inplace=True)
df.to_pickle('data_list_df.pkl')