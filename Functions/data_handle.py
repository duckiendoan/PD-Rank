# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 16:43:31 2021

@author: filipavaldeira
"""

import timeit
import numpy as np
import networkx as nx

from random import sample
from scipy.sparse import csr_matrix
from itertools import combinations


##############################################################################
##### Generate data
##############################################################################


class Data(object):

    def __init__(self, m, n, gt = None, 
                 noise_type = 'constant', noise_level = 0.1, bt_interval = [0,5],
                 connectivity = 'connected', demo=True):
        
        """ 
            Parameters
            ----------
            m : int
                number of items
                
            n : int
                number of comparisons
                
            gt : array-like, optional
                Ground truth ranking of item ids. If not specified, ground truth is a permutation of 1:n.
            
            noise_type : str, optional
                Noise model. Options are 'constant', 'BTL', or 'None'. Default is 'constant'.
            
            noise_level : float, optional
                Level of noise to be added. Default is 0.1.
                   
            bt_interval : list, optional
                Interval for generating BTL noise. Default is [0, 5].
            
            connectivity : str, optional
                Connectivity type. Options are 'no', 'connected', or 'strongly'. Default is 'connected'.
    
        """ 
        if demo:
            print(f'Data generation options:')
            print(f'- Noise: {noise_type}')
            print(f'- Number of items: {m}')
            print(f'- Number of comparisons: {n}')
            print(f'- Bradley-Terry interval: {bt_interval}')
            print(f'- Connectivity: {connectivity}')
        
        
        # Ground truth
        if gt is None:
            if noise_type == 'BTL':
                x = np.random.uniform(bt_interval[0],bt_interval[1],size= m)# ranking
            else:
                x = np.random.permutation(range(m))
        else:
            x = gt
        
        
        pairs = self._select_pairs(m,n)
        pairs.sort(axis=1)  
        diff = x[pairs][:,0]-x[pairs][:,1]
        y_true = np.sign(diff).flatten()
        
        delta = genNoise(noise_type,noise_level,x,pairs)
        y_noise = gen_y_obs(y_true,delta)  

        pairs_noise = pairs.copy()
        pairs_noise[y_noise==-1] = np.flip(pairs_noise[y_noise==-1],axis=1)
        
        unq_pairs, ids_all, tt_obs = np.unique(pairs_noise, return_inverse=True, return_counts=True, axis=0)        
     
        self.P_abs, self.x = self._pairs_to_P(unq_pairs,tt_obs,connectivity,x)
        self.m, self.n = self.P_abs.shape[0], self.P_abs.sum()
        self.y_true,self.y_noise = y_true, y_noise
        self.ids_unq = ids_all
        
        
    #  ---- Get data in different formats ------ #
    def get_rankings(self):
        pairs, w = self._get_unq_pairs_w()
        rankings = np.repeat(pairs, w, axis=0)
        return rankings
        
    
    def get_Areduced_w(self):
        pairs, w = self._get_unq_pairs_w()
        m = self.m
        n_unq = pairs.shape[0]
        
        vals = np.concatenate((np.ones(n_unq),-1*np.ones(n_unq)),axis=0)
        ids_row = np.concatenate((np.arange(n_unq),np.arange(n_unq)),axis=0)
        ids_col = np.concatenate((pairs[:,0],pairs[:,1]),axis=0)
   
        A = csr_matrix((vals,(ids_row,ids_col)),shape = (n_unq,m))
        return A, w

    def get_A_full(self):
        # Get the full A matrix
        rankings = self.get_rankings()
        n_full = rankings.shape[0]
        vals = np.concatenate((np.ones(n_full),-1*np.ones(n_full)),axis=0)
        ids_row = np.concatenate((np.arange(n_full),np.arange(n_full)),axis=0)
        ids_col = np.concatenate((rankings[:,0],rankings[:,1]),axis=0)

        A_full = csr_matrix((vals,(ids_row,ids_col)),shape = (n_full, self.m))
        return A_full
    
    def get_P_rel(self):
        P_sum = self.P_abs+self.P_abs.transpose()
        P_sum.data = 1/P_sum.data
        P_rel = self.P_abs.multiply(P_sum)
        return P_rel

    def get_P_abs(self):
        return self.P_abs
    
    #  ---- Labels ------ #
    def get_y(self,x):
        rankings = self.get_rankings()
        diff = x[rankings][:,0]-x[rankings][:,1]
        y = np.sign(diff).flatten()
        return y
    
    def get_y_red(self,x):
        pairs, w = self._get_unq_pairs_w()
        diff = x[pairs][:,0]-x[pairs][:,1]
        y = np.sign(diff).flatten()
        return y
    

    def _get_unq_pairs_w(self):
        x,y = self.P_abs.nonzero()
        w = self.P_abs.data
        pairs = np.concatenate((x.reshape(-1,1),y.reshape(-1,1)),axis=1)
        return pairs, w        
    
    def _select_pairs(self,m,n):
        pairs = pick_pairs(m,n)
        pairs.sort(axis=1)                    

        return pairs
    
    def _pairs_to_P(self,unq_pairs,tt_obs,connectivity,x):
        
        G = nx.DiGraph()
        G.add_nodes_from(range(len(x)))
        G.add_weighted_edges_from(np.concatenate((unq_pairs,tt_obs.reshape(-1,1)),axis=1))       
        
        if connectivity == 'no':
            print('Not checking connectivity')
            G_connected = G.copy()
            largest =G_connected.nodes()
        else:
            if connectivity == 'connected':
                # print('Checking connected components')
                tic = timeit.default_timer()
                largest = max(nx.weakly_connected_components(G), key=len)
            elif connectivity == 'strongly':
                print('Checking strongly connected components')                
                largest = max(nx.strongly_connected_components(G), key=len)
                
            G_connected = G.subgraph(largest).copy()

        tic = timeit.default_timer()
        P_abs = nx.adjacency_matrix(G_connected)
        ids_keep = np.array(list(largest)).astype(int)
        x_trim = x[ids_keep]

        return P_abs,x_trim

     
##############################################################################
##### Aux functions
##############################################################################

def pick_pairs(m,n):
    r_init = np.random.choice(m,(n,2), replace=True)
    equal = r_init[:,0]==r_init[:,1]
    n_miss = np.sum(equal)
    if n_miss>0:
        r_add = np.array([np.random.choice(m,2, replace=False) for i in np.arange(n_miss) ])
        r = np.concatenate((r_init[~equal],r_add),axis=0)
    else:
        r = r_init
    
    return r

def genNoise(kind,level,x_gt,pairs):
    
    n = pairs.shape[0]
    
    if kind == 'constant':
        delta = np.ones(n)*level
    elif kind == 'None':
        delta = None
    elif kind == 'BTL':
        diff = abs(np.diff(x_gt[pairs],axis=1))
        delta= np.exp(-diff)/(1+np.exp(-diff))
    elif kind == 'random_annotator':
        # Assign an annotator to each pair with uniform probability
        # Here level is a numpy array, indicating a different noise level for each annotator
        n_annotator = level.size
        delta = np.random.choice(level, size=(n,))
    else:
        raise NotImplementedError('Error: noise option not available! \n Available options: constant, none, BTL')

    if delta is not None:
        delta = delta.flatten()
    return delta

def gen_y_obs(y_true,delta):
    if delta is None:
        y_noise = y_true
    else :
        n = y_true.shape[0]
        noise_proba = np.random.rand(n)
        z = np.ones(n)
        z[noise_proba<delta] = -1 
        y_noise = y_true * z
    return y_noise
