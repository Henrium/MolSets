#%%
import pickle
from torch_geometric.data import Dataset, Batch, Data
from torch_geometric.data.data import size_repr
import torch
import numpy as np


class GraphSetDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        with open(data_path, 'rb') as f:
            self.data_list = pickle.load(f)
        
        targets = np.array([self.data_list[i][-1][0] for i in range(len(self.data_list))])
        self.target_mean = np.mean(targets)
        self.target_std = np.std(targets)
    
    def len(self):
        return len(self.data_list)
    
    def get(self, idx):
        sample = self.data_list[idx]
        normalized_target = [float((sample[-1][0] - self.target_mean) / self.target_std)]
        normalized_sample = sample[:-1] + (normalized_target,)
        return normalized_sample

    def get_orig(self, target):
        return target * self.target_std + self.target_mean


def graph_set_collate(batch):
    _, graph_lists, mws, weight_lists, mol_lists, salt_graphs, ys = zip(*batch)
    batched_graph_sets = [Batch.from_data_list(g_list) for g_list in graph_lists]
    return batched_graph_sets, mws, weight_lists, mol_lists, salt_graphs, ys

# Data classes for DMPNN
class RevIndexedData(Data):
    def __init__(self, orig):
        super(RevIndexedData, self).__init__()
        if orig:
            for key in orig.keys():
                self[key] = orig[key]
            edge_index = self["edge_index"]
            revedge_index = torch.zeros(edge_index.shape[1]).long()
            for k, (i, j) in enumerate(zip(*edge_index)):
                edge_to_i = edge_index[1] == i
                edge_from_j = edge_index[0] == j
                revedge_index[k] = torch.where(edge_to_i & edge_from_j)[0].item()
            self["revedge_index"] = revedge_index

    def __inc__(self, key, value, *args, **kwargs):
        if key == "revedge_index":
            return self.revedge_index.max().item() + 1
        else:
            return super().__inc__(key, value)

    def __repr__(self):
        cls = str(self.__class__.__name__)
        has_dict = any([isinstance(item, dict) for _, item in self])

        if not has_dict:
            info = [size_repr(key, item) for key, item in self]
            return "{}({})".format(cls, ", ".join(info))
        else:
            info = [size_repr(key, item, indent=2) for key, item in self]
            return "{}(\n{}\n)".format(cls, ",\n".join(info))
        
class RevIndexedDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        with open(data_path, 'rb') as f:
            orig_list = pickle.load(f)
        self.data_list = []
        for sample in orig_list:
            graph_list = sample[1]
            updated_graph_list = []
            for graph in graph_list:
                updated_graph = RevIndexedData(graph)
                updated_graph_list.append(updated_graph)
            updated_sample = (sample[0], updated_graph_list, sample[2], sample[3], sample[4], RevIndexedData(sample[5]), sample[6])
            self.data_list.append(updated_sample)
        
        targets = np.array([self.data_list[i][-1][0] for i in range(len(self.data_list))])
        self.target_mean = np.mean(targets)
        self.target_std = np.std(targets)
    
    def len(self):
        return len(self.data_list)
    
    def get(self, idx):
        sample = self.data_list[idx]
        normalized_target = [float((sample[-1][0] - self.target_mean) / self.target_std)]
        normalized_sample = sample[:-1] + (normalized_target,)
        return normalized_sample

    def get_orig(self, target):
        return target * self.target_std + self.target_mean