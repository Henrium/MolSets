from data_utils import GraphSetDataset
import pandas as pd
import torch
from torch_geometric.data import Batch
from models import MolSets
import numpy as np

# Candidate mixtures to be predicted
candidate_data = pd.read_pickle('./data/all_bin_candidates.pkl')
# Dataset where the model is trained
train_ds = GraphSetDataset('./data/data_list.pkl')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MolSets(n_node_features=13, hidden_dim=16, emb_dim=32, att_dim=16, output_dim=1, conv='GraphConv', n_conv_layers=3, after_readout='tanh').to(device)
# Load trained model
model.load_state_dict(torch.load('results/GraphConv_3_h16_e32_att16_tanh.pt'))

predicted = []

for i in range(len(candidate_data)):
    graph_list, mw, frac, salt_mol, salt_graph = candidate_data.iloc[i]
    graph_list = Batch.from_data_list(graph_list).to(device)
    salt_graph = salt_graph.to(device)
    frac = torch.tensor(frac, dtype=torch.float).to(device)
    salt_mol = torch.tensor(salt_mol).to(device)
    mw = torch.tensor(mw).to(device)
    
    predicted.append(model(graph_list, mw, frac, salt_mol, salt_graph).item())

    if i % 100 == 0:
        print('Finished {} samples'.format(i))

preds = train_ds.get_orig(np.array(predicted))
candidate_data['predicted'] = preds

candidate_data.to_csv('candidate_data_pred.csv')
