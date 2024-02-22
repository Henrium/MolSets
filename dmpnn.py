import torch
from torch_geometric.nn.pool import global_mean_pool
import torch.nn.functional as F
from torch_scatter import scatter_sum

# DMPNN conv layer
# Source: https://github.com/itakigawa/pyg_chemprop
def directed_mp(message, edge_index, revedge_index):
    m = scatter_sum(message, edge_index[1], dim=0)
    m_all = m[edge_index[0]]
    m_rev = message[revedge_index]
    return m_all - m_rev

def aggregate_at_nodes(num_nodes, message, edge_index):
    m = scatter_sum(message, edge_index[1], dim=0, dim_size=num_nodes)
    return m[torch.arange(num_nodes)]


class DMPNNEncoder(torch.nn.Module):
    def __init__(self, hidden_size, emb_dim, node_fdim, edge_fdim, after_readout, depth=3, input_type=None):
        super(DMPNNEncoder, self).__init__()
        self.act_func = torch.nn.ReLU()
        self.after_readout = after_readout
        self.W1 = torch.nn.Linear(node_fdim + edge_fdim, hidden_size, bias=False)
        self.W2 = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.W3 = torch.nn.Linear(node_fdim + hidden_size, hidden_size, bias=True)
        self.depth = depth
        
        self.fc = torch.nn.Linear(hidden_size, emb_dim) if input_type == 'nomw' else torch.nn.Linear(hidden_size+1, emb_dim)
        self.tanh = torch.nn.Tanh()

    def forward(self, data, logmw=None):
        x, edge_index, revedge_index, edge_attr, num_nodes, batch = (
            data.x,
            data.edge_index,
            data.revedge_index,
            data.edge_attr.unsqueeze(1),
            data.num_nodes,
            data.batch,
        )

        # initialize messages on edges
        init_msg = torch.cat([x[edge_index[0]], edge_attr], dim=1).float()
        h0 = self.act_func(self.W1(init_msg))

        # directed message passing over edges
        h = h0
        for _ in range(self.depth - 1):
            m = directed_mp(h, edge_index, revedge_index)
            h = self.act_func(h0 + self.W2(m))

        # aggregate in-edge messages at nodes
        v_msg = aggregate_at_nodes(num_nodes, h, edge_index)

        z = torch.cat([x, v_msg], dim=1)
        node_attr = self.act_func(self.W3(z))

        # readout: pyg global pooling
        x = global_mean_pool(node_attr, batch)
        if logmw is not None:
            x = torch.cat((x, logmw), dim=1)
        match self.after_readout:
            case 'tanh': x = self.tanh(self.fc(x))
            case 'norm': x = F.normalize(self.fc(x), dim=1)
            case _: raise NotImplementedError

        return x


class MolSets_DMPNN(torch.nn.Module):
    def __init__(self, n_node_features, hidden_dim, emb_dim, output_dim, n_conv_layers, after_readout, activation='relu'):
        super().__init__()
        self.hidden_dim = hidden_dim
        match activation:
            case 'relu': self.act = torch.nn.ReLU()
            case 'silu': self.act = torch.nn.SiLU()
            case 'gelu': self.act = torch.nn.GELU()
            case 'lrelu': self.act = torch.nn.LeakyReLU()
            case _: raise NotImplementedError
        self.phi = DMPNNEncoder(hidden_dim, emb_dim, n_node_features, edge_fdim=1, after_readout=after_readout, depth=n_conv_layers)
        self.phi_salt = DMPNNEncoder(hidden_dim, emb_dim, n_node_features, edge_fdim=1, after_readout=after_readout, depth=n_conv_layers, input_type='nomw')
        self.att_q_net = torch.nn.Linear(emb_dim, 16)
        self.att_k_net = torch.nn.Linear(emb_dim, 16)
        self.att_v_net = torch.nn.Linear(emb_dim, emb_dim)
        self.rho = torch.nn.Sequential(
            torch.nn.Linear(2*emb_dim+1, hidden_dim),
            self.act,
            torch.nn.Linear(hidden_dim, hidden_dim),
            self.act,
            torch.nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, graph_list, mw, frac, salt_mol, salt_graph):
        log_mw = torch.log10(mw).unsqueeze(1)
        # graph_list is a batch of graphs in one datapoint
        embeddings = self.phi(graph_list, log_mw)  # n_graphs * emb_dim

        salt_embedding = self.phi_salt(salt_graph).squeeze()
        # Permutation-invariant aggregation using attention mechanism
        att_queries = self.att_q_net(embeddings)  # n_graphs * att_dim
        att_values = self.att_v_net(embeddings)  # n_graphs * emb_dim
        att_keys = self.att_k_net(embeddings)  # n_graphs * att_dim
        att_scores = torch.matmul(att_queries, att_keys.transpose(0, 1)) / (att_keys.size(1) ** 0.5) # n_graphs * n_graphs
        att_outputs = torch.matmul(torch.softmax(att_scores, dim=0), att_values) # n_graphs * emb_dim
        x = torch.matmul(frac, att_outputs).squeeze()
        # Representation of polymer mixture
        x = torch.cat((x, salt_embedding, salt_mol.unsqueeze(0)))
        x = self.rho(x)
        return x
