import timeit
import numpy as np
import networkx as nx

import random
from scipy.sparse import csr_matrix

def sample_pairs(m,n):
    r_init = np.random.choice(m,(n,2), replace=True)
    equal = r_init[:,0]==r_init[:,1]
    n_miss = np.sum(equal)
    if n_miss>0:
        r_add = np.array([np.random.choice(m,2, replace=False) for i in np.arange(n_miss) ])
        r = np.concatenate((r_init[~equal],r_add),axis=0)
    else:
        r = r_init

    return r

def sample_pairs_no_duplicate(n, k):
    if n < 2:
        raise ValueError("n must be at least 2")
    total_pairs = n * (n - 1)
    if k > total_pairs:
        raise ValueError("k cannot exceed the total number of possible pairs")
    
    # Sample k unique indices from the range of possible pair indices
    indices = random.sample(range(total_pairs), k)
    pairs = []
    for idx in indices:
        # Determine i and j from the linear index
        i = idx // (n - 1)
        j = idx % (n - 1)
        # Adjust j to skip the diagonal where i == j
        if j >= i:
            j += 1
        pairs.append((i, j))
    return pairs

def generate_noise(noise_type, noise_level, x, pairs):
    n = pairs.shape[0]

    if noise_type == 'random_annotator':
        noise_level = np.asarray(noise_level)
        annotators = np.random.choice(noise_level.shape[0], n)
    else:
        annotators = np.zeros(n, dtype=int)

    if noise_type == 'constant':
        delta = np.ones(n) * noise_level
    elif noise_type == 'None':
        delta = None
    elif noise_type == 'BTL':
        diff = abs(np.diff(x[pairs],axis=1))
        delta = np.exp(-diff)/(1+np.exp(-diff))
    elif noise_type == 'random_annotator':
        delta = noise_level[annotators]
    else:
        raise NotImplementedError('Error: noise option not available! \n Available options: constant, none, BTL')

    if delta is not None:
        delta = delta.flatten()

    return annotators, delta

def apply_noise(y_true, noise):
    if noise is not None:
        n = y_true.shape[0]
        noise_proba = np.random.rand(n)
        z = np.ones(n)
        z[noise_proba < noise] = -1
        y_noise = y_true * z
        return y_noise
    return y_true


class DataExp:
    def __init__(self, m, n, gt = None, 
                 noise_type = 'constant', noise_level = 0.1, bt_interval = [0,5],
                 connectivity = 'connected', demo=True):
        if gt is None:
            if noise_type == 'BTL':
                x = np.random.uniform(bt_interval[0], bt_interval[1], size=m)# ranking
            else:
                x = np.random.permutation(range(m))
        else:
            x = gt

        # Generate data
        pairs = sample_pairs(m, n)
        y = np.sign(-np.diff(x[pairs], axis=1)).flatten()
        # Add noise
        annotators, noise = generate_noise(noise_type, noise_level, x, pairs)
        y_noise = apply_noise(y, noise)
        # Swap so that the item with higher score goes first -> no need to store y
        pairs[y_noise == -1] = np.flip(pairs[y_noise == -1], axis=1)
        # Store annotators for each observation
        annotator_map = {}
        for i in range(pairs.shape[0]):
            annotator_map[tuple(pairs[i])] = annotator_map.get(tuple(pairs[i]), []) + [annotators[i]]
        # Reduced form. To recover: np.repeat(pairs_reduced, pair_counts, axis=0)
        pairs_reduced, pair_counts = np.unique(pairs, return_counts=True, axis=0)
        # Graph form (also ensure connectivity)
        ids_keep, adj_matrix, x_new = self.build_graph(pairs_reduced, pair_counts, connectivity, x)
        new_m, new_n = adj_matrix.shape[0], int(adj_matrix.sum())
        new_pairs, new_weight = self.adj_matrix_to_pairs(adj_matrix)
        # If something changes, we need to rebuild annotator map
        if new_m != m or new_n != n:
            new_annotator_map = {}
            for pair in new_pairs:
                old_idx = (np.int64(ids_keep[pair[0]]), np.int64(ids_keep[pair[1]]))
                new_annotator_map[tuple(pair)] = annotator_map[old_idx]

            for pair, w in zip(new_pairs, new_weight):
                assert len(new_annotator_map[tuple(pair)]) == w
        else:
            new_annotator_map = annotator_map
        
        # Store everything:
        self.m, self.n = new_m, new_n
        self.x = x_new
        self.pairs_reduced = new_pairs
        self.pairs_weight = new_weight
        self.annotator_map = new_annotator_map
    
    def get_Areduced_w(self):
        pairs, w = self.pairs_reduced, self.pairs_weight
        m = self.m
        n_unq = pairs.shape[0]
        
        vals = np.concatenate((np.ones(n_unq),-1*np.ones(n_unq)),axis=0)
        ids_row = np.concatenate((np.arange(n_unq),np.arange(n_unq)),axis=0)
        ids_col = np.concatenate((pairs[:,0],pairs[:,1]),axis=0)
   
        A = csr_matrix((vals,(ids_row,ids_col)),shape = (n_unq,m))
        return A, w

    def get_A_full(self, return_annotators=False):
        # Get the full A matrix
        all_pairs = np.repeat(self.pairs_reduced, self.pairs_weight, axis=0)

        n_full = all_pairs.shape[0]
        vals = np.concatenate((np.ones(n_full),-1*np.ones(n_full)),axis=0)
        ids_row = np.concatenate((np.arange(n_full),np.arange(n_full)),axis=0)
        ids_col = np.concatenate((all_pairs[:,0],all_pairs[:,1]),axis=0)

        A_full = csr_matrix((vals,(ids_row,ids_col)),shape = (n_full, self.m))

        if return_annotators:
            all_annotators = np.concat([self.annotator_map[tuple(pair)] for pair in self.pairs_reduced])
            return A_full, all_annotators
            
        return A_full


    def build_graph(self, pairs, pairs_weight, connectivity, x):
        G = nx.DiGraph()
        G.add_nodes_from(range(len(x)))
        G.add_weighted_edges_from(np.concatenate((pairs, pairs_weight.reshape(-1,1)),axis=1))

        if connectivity == 'no':
            print('Not checking connectivity')
            G_connected = G.copy()
            largest = G_connected.nodes()
        else:
            if connectivity == 'connected':
                # print('Checking connected components')
                tic = timeit.default_timer()
                largest = max(nx.weakly_connected_components(G), key=len)
            elif connectivity == 'strongly':
                # print('Checking strongly connected components')
                largest = max(nx.strongly_connected_components(G), key=len)

            G_connected = G.subgraph(largest).copy()

        tic = timeit.default_timer()
        P_abs = nx.adjacency_matrix(G_connected)
        ids_keep = np.array(list(G_connected.nodes)).astype(int)
        x_trim = x[ids_keep]

        return ids_keep, P_abs, x_trim

    def adj_matrix_to_pairs(self, adj_matrix):
        x, y = adj_matrix.nonzero()
        w = adj_matrix.data
        pairs = np.concatenate((x.reshape(-1,1),y.reshape(-1,1)),axis=1)
        return pairs, w    