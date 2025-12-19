import networkx as nx
import torch

def generate_graph(n=10):
    G = nx.erdos_renyi_graph(n, 0.3)
    x = torch.eye(n)  # node features
    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    if edge_index.numel() == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    return G, x, edge_index
