import math
import pickle
import operator
import dgl
import dgl.data
import os
import numpy as np
import torch
import multiprocessing
from time import time
import psutil
from scipy.sparse import linalg
import scipy.sparse as sparse
import sklearn.preprocessing as preprocessing
import torch.nn.functional as F


def _rwr_trace_to_dgl_graph(
    g, seed, trace, positional_embedding_size, entire_graph=False
):
    subv = torch.unique(torch.cat(trace)).tolist()
    try:
        subv.remove(seed)
    except ValueError:
        pass
    subv = [seed] + subv
    if entire_graph:
        subg = g.subgraph(g.nodes())
    else:
        # use sample nodes to generate subgraph   and Anonymization nodes, the center node idx is 0
        subg = g.subgraph(subv)

    subg = _add_undirected_graph_positional_embedding(
        subg, positional_embedding_size)

    subg.ndata["seed"] = torch.zeros(subg.number_of_nodes(), dtype=torch.long)
    if entire_graph:
        subg.ndata["seed"][seed] = 1
    else:
        subg.ndata["seed"][0] = 1
    return subg


def eigen_decomposision(n, k, laplacian, hidden_size, retry):
    if k <= 0:
        return torch.zeros(n, hidden_size)
    laplacian = laplacian.astype("float64")
    ncv = min(n, max(2 * k + 1, 20))
    # follows https://stackoverflow.com/questions/52386942/scipy-sparse-linalg-eigsh-with-fixed-seed
    v0 = np.random.rand(n).astype("float64")
            for i in range(retry):
        try:
            s, u = linalg.eigsh(laplacian, k=k, which="LA", ncv=ncv, v0=v0)
        except sparse.linalg.eigen.arpack.ArpackError:
            # print("arpack error, retry=", i)
            ncv = min(ncv * 2, n)
            if i + 1 == retry:
                sparse.save_npz("arpack_error_sparse_matrix.npz", laplacian)
                u = torch.zeros(n, k)
        else:
            break
    x = preprocessing.normalize(u, norm="l2")
    x = torch.from_numpy(x.astype("float32"))
    x = F.pad(x, (0, hidden_size - k), "constant", 0)
    return x


def _add_undirected_graph_positional_embedding(g, hidden_size, retry=10):
    n = g.number_of_nodes()
    adj = g.adjacency_matrix_scipy(
        transpose=False, return_edge_ids=False).astype(float)
    norm = sparse.diags(dgl.backend.asnumpy(
        g.in_degrees()).clip(1) ** -0.5, dtype=float)
    laplacian = norm * adj * norm
    k = min(n - 2, hidden_size)
    x = eigen_decomposision(n, k, laplacian, hidden_size, retry)
    g.ndata["pos_undirected"] = x.float()
    return g


class Dataset():
    def __init__(
        self,
        rw_hops=64,
        subgraph_size=64,
        restart_prob=0.8,
        positional_embedding_size=32,
        step_dist=[1.0, 0.0, 0.0],
    ):
        self.rw_hops = rw_hops
        self.subgraph_size = subgraph_size
        self.restart_prob = restart_prob
        self.positional_embedding_size = positional_embedding_size
        self.step_dist = step_dist
        assert sum(step_dist) == 1.0
        assert positional_embedding_size > 1

        self.dgl_graphs_file = "yelp2018.train.bin"
        self.graphs, _ = dgl.data.utils.load_graphs(self.dgl_graphs_file)
        self.nums = 2000

    def __update__(self):
        print("Node traces updating")
        # traces = dgl.contrib.sampling.random_walk_with_restart(
        #     self.graphs[0],
        #     seeds=np.arange(self.item_max + 1),
        #     restart_prob=self.restart_prob,
        #     max_nodes_per_seed=self.rw_hops,
        # )
        # self.node_subgraph = [data_util._rwr_trace_to_dgl_graph(
        #     g=self.graphs[0],
        #     seed=i,
        #     trace=traces[i],
        #     positional_embedding_size=self.positional_embedding_size,
        # )
        #     for i in trange(self.item_max + 1)
        # ]
        time1 = time()
        pool = multiprocessing.Pool(10)
        self.node_subgraph = pool.map(
            self.get_subgraph, torch.tensor(np.arange(self.nums)))
        time2 = time()
        pool.close()
        pool.join()
        print(time2-time1)

    def get_subgraph(self, i):
        print(i)
        trace = dgl.contrib.sampling.random_walk_with_restart(
            self.graphs[0],
            seeds=[i],
            restart_prob=self.restart_prob,
            max_nodes_per_seed=self.rw_hops,
        )
        node_subgraph = _rwr_trace_to_dgl_graph(
            g=self.graphs[0],
            seed=i,
            trace=trace[0],
            positional_embedding_size=self.positional_embedding_size,
        )
        return node_subgraph


if __name__ == "__main__":
    dataset = Dataset()
    dataset.__update__()
