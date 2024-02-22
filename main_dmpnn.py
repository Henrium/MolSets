import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import random_split
import numpy as np
from dmpnn import MolSets_DMPNN
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
import random
from scipy.stats import spearmanr, pearsonr
from data_utils import  graph_set_collate, RevIndexedDataset

hyperpars = {
    # Architecture
    'hidden_dim': 32,
    'emb_dim': 16,
    'n_conv_layers': 3,
    'after_readout': 'tanh',
    # Training
    'max_ep': 10000,
    'es_patience': 10,
    'max_ep_wo_improv': 20,
    # Learning rate
    'lr': 0.001,
    'lrsch_patience': 10,
    'lrsch_factor': 0.5,
    # Regularization
    'weight_decay': 0.0001
}

best_model = None
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

dataset = RevIndexedDataset('./data/data_list.pkl')
train_data, val_data, test_data = random_split(dataset, (0.6, 0.2, 0.2))
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
train_loader.collate_fn = graph_set_collate
val_loader = DataLoader(val_data, batch_size=32, shuffle=True)
val_loader.collate_fn = graph_set_collate

# Train model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MolSets_DMPNN(n_node_features=13, hidden_dim=hyperpars['hidden_dim'], emb_dim=hyperpars['emb_dim'], output_dim=1, n_conv_layers=hyperpars['n_conv_layers'], after_readout=hyperpars['after_readout']).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=hyperpars['lr'], weight_decay=hyperpars['weight_decay'])
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=hyperpars['lrsch_factor'], patience=hyperpars['lrsch_patience'], verbose=True)
loss_fn = torch.nn.MSELoss()

def train(model, loader, optimizer, criterion):
    model.train()
    train_loss = 0
    for sample in loader:
        # "inputs": a batch of graphs sets
        inputs, mws, fracs, salt_mols, salt_graphs, targets = sample
        sample_size = len(targets)
        outs = torch.empty((sample_size, 1)).to(device)        
        targets = torch.tensor(targets).to(device)
        salt_mols = torch.tensor(salt_mols).to(device)
        for j in range(sample_size):   
            graph_set = inputs[j].to(device)
            salt_graph = salt_graphs[j].to(device)
            frac = torch.tensor(fracs[j]).to(device)
            mw = torch.tensor(mws[j]).to(device)
            optimizer.zero_grad()
            outs[j] = model(graph_set, mw, frac, salt_mols[j], salt_graph)
        loss = criterion(outs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(train_loader)

def evaluate(model, loader, criterion):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for sample in loader:
            inputs, mws, fracs, salt_mols, salt_graphs, targets = sample
            sample_size = len(targets)
            outs = torch.empty((sample_size, 1)).to(device)
            targets = torch.tensor(targets).to(device)
            salt_mols = torch.tensor(salt_mols).to(device)
            for j in range(sample_size):
                graph_set = inputs[j].to(device)
                salt_graph = salt_graphs[j].to(device)
                frac = torch.tensor(fracs[j]).to(device)
                mw = torch.tensor(mws[j]).to(device)
                outs[j] = model(graph_set, mw, frac, salt_mols[j], salt_graph)
            loss = criterion(outs, targets)
            val_loss += loss.item()
    return val_loss / len(val_loader)

# Set early stopping criteria
best_val_loss = np.inf
epochs_wo_improv = 0
print(f'Total params: {sum(param.numel() for param in model.parameters())}')

# The training loop
for epoch in range(hyperpars['max_ep']):
    train_loss = train(model, train_loader, optimizer, loss_fn)
    val_loss = evaluate(model, val_loader, loss_fn)    
    scheduler.step(val_loss)

    # Early stopping check
    if epoch > hyperpars['es_patience']:
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
            epochs_wo_improv = 0
        else:
            epochs_wo_improv += 1
        if epochs_wo_improv >= hyperpars['max_ep_wo_improv']:
            print(f'Early stopping at epoch {epoch+1}')
            break

    print(f'Epoch {epoch+1}: Train Loss={train_loss:.5f}, Val Loss={val_loss:.5f}')

#%% Plots
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
mol_types = pd.read_pickle('./data/data_df_stats.pkl')['mol_type']
model_name = 'DMPNN_{}_h{}_e{}_reg{}_{}'.format(hyperpars['n_conv_layers'], hyperpars['hidden_dim'], hyperpars['emb_dim'], hyperpars['weight_decay'], hyperpars['after_readout'])

targets = []
predicted = []
mol_labels = []
mol_types_list = []

if best_model is not None:
    model.load_state_dict(best_model)
    torch.save(best_model, 'results/{}.pt'.format(model_name))

model.eval()
with torch.no_grad():
    for sample in test_data:
        index, inputs, mw, frac, salt_mol, salt_graph, target = sample
        inputs = Batch.from_data_list(inputs).to(device)
        target = torch.tensor(target).to(device)
        frac = torch.tensor(frac).to(device)
        salt_mol = torch.tensor(salt_mol).to(device)
        mw = torch.tensor(mw).to(device)
        salt_graph.to(device)
        out = model(inputs, mw, frac, salt_mol, salt_graph)
        targets.append(target.cpu().numpy())
        predicted.append(out.cpu().numpy())
        mol_types_list.append(mol_types[index])
        match mol_types[index]:
            case 'poly': mol_labels.append(2)
            case 'mixed': mol_labels.append(1)
            case 'small': mol_labels.append(0)

targets = dataset.get_orig(np.stack(targets).squeeze())
predicted = dataset.get_orig(np.stack(predicted).squeeze())

results = pd.DataFrame({'target': targets, 'predicted': predicted, 'mix_type': mol_types_list})

spearman_r = spearmanr(targets, predicted)
pearson_r = pearsonr(targets, predicted)

print('Spearman r: {}, Pearson r: {}'.format(spearman_r, pearson_r))

sns.scatterplot(data=results, x='target', y='predicted', hue='mix_type')
plt.gca().set_aspect('equal', adjustable='box')
plt.axline([0, 0], [1, 1], color='black')
plt.xlabel('Target - log(S/cm)')
plt.ylabel('Predicted - log(S/cm)')
plt.savefig('results/{}.png'.format(model_name))

results.to_csv('results/{}.csv'.format(model_name), index=False)
