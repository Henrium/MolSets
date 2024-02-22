import torch
from torch_geometric.nn.conv import GraphConv, GCNConv, GATConv, SAGEConv
from torch_geometric.nn.norm import LayerNorm
from torch_geometric.nn.pool import global_mean_pool
import torch.nn.functional as F

class GraphEmbedding(torch.nn.Module):
    def __init__(self, n_node_features, hidden_dim, emb_dim, conv, n_conv_layers, after_readout, activation, input_type=None):
        super(GraphEmbedding, self).__init__()
        match conv:      
            case 'SAGEConv': ConvLayer = SAGEConv
            case 'GCNConv': ConvLayer = GCNConv
            case 'GATConv': ConvLayer = GATConv
            case 'GraphConv': ConvLayer = GraphConv
            case _: raise NotImplementedError
        self.act = activation
        self.after_readout = after_readout
        self.convs = torch.nn.ModuleList()
        self.convs.append(ConvLayer(n_node_features, hidden_dim))
        for _ in range(n_conv_layers-1):
            self.convs.append(ConvLayer(hidden_dim, hidden_dim))
        self.norm = LayerNorm(hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, emb_dim) if input_type == 'nomw' else torch.nn.Linear(hidden_dim+1, emb_dim)
        self.tanh = torch.nn.Tanh()
        
    def forward(self, graph, logmw=None):
        x, edge_index, edge_attr = graph.x, graph.edge_index, graph.edge_attr.unsqueeze(1)
        for conv in self.convs:
            try:
                x = conv(x=x, edge_index=edge_index, edge_weight=edge_attr)
            except TypeError:
                x = conv(x=x, edge_index=edge_index)
            x = self.norm(x)
            x = self.act(x)
        x = global_mean_pool(x, graph.batch)
        if logmw is not None:
            x = torch.cat((x, logmw), dim=1)
        match self.after_readout:
            case 'tanh': x = self.tanh(self.fc(x))
            case 'norm': x = F.normalize(self.fc(x), dim=1)
            case _: raise NotImplementedError
        return x

class MolSets(torch.nn.Module):
    def __init__(self, n_node_features, hidden_dim, emb_dim, att_dim, output_dim, conv, n_conv_layers, after_readout='tanh', activation='relu'):
        super(MolSets, self).__init__()
        self.hidden_dim = hidden_dim
        match activation:
            case 'relu': self.act = torch.nn.ReLU()
            case 'silu': self.act = torch.nn.SiLU()
            case 'gelu': self.act = torch.nn.GELU()
            case 'lrelu': self.act = torch.nn.LeakyReLU()
            case _: raise NotImplementedError

        self.phi = GraphEmbedding(n_node_features, hidden_dim, emb_dim, conv, n_conv_layers, after_readout, self.act)
        self.phi_salt = GraphEmbedding(n_node_features, hidden_dim, emb_dim, conv, n_conv_layers, after_readout, self.act, 'nomw')
        self.att_q_net = torch.nn.Linear(emb_dim, att_dim)
        self.att_k_net = torch.nn.Linear(emb_dim, att_dim)
        self.att_v_net = torch.nn.Linear(emb_dim, emb_dim)
        self.rho = torch.nn.Sequential(
            torch.nn.Linear(2*emb_dim+1, hidden_dim),
            self.act,
            torch.nn.Linear(hidden_dim, hidden_dim),
            self.act,
            torch.nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, graph_list, mw, frac, salt_mol, salt_graph):
        # graph_list is a batch of graphs in one datapoint
        log_mw = torch.log10(mw).unsqueeze(1)
        embeddings = self.phi(graph_list, log_mw)  # n_graphs * emb_dim

        salt_embedding = self.phi_salt(salt_graph).squeeze()
        # Permutation-invariant aggregation using attention mechanism
        att_queries = self.att_q_net(embeddings)  # n_graphs * att_dim
        att_values = self.att_v_net(embeddings)  # n_graphs * emb_dim
        att_keys = self.att_k_net(embeddings)  # n_graphs * att_dim
        att_scores = torch.matmul(att_queries, att_keys.transpose(0, 1)) / (att_keys.size(1) ** 0.5) # n_graphs * n_graphs
        att_outputs = torch.matmul(torch.softmax(att_scores, dim=0), att_values) # n_graphs * emb_dim
        x = torch.matmul(frac, att_outputs).squeeze()
        # Representation of a mixture
        x = torch.cat((x, salt_embedding, salt_mol.unsqueeze(0)))
        x = self.rho(x)
        return x
