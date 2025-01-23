# -*- coding: utf-8 -*-

##############################################################################
# Imports
##############################################################################

import sys
from os.path import dirname, realpath, join
import numpy as np

project_dir = dirname(dirname(dirname(dirname(realpath(__file__)))))
sys.path.append(join(project_dir, 'Code'))

from Functions.method import PDRANK
from Functions.data_handle import Data

##############################################################################
# Settings
##############################################################################

m = 10 # number of items
n = 500 # number of comparisons

# Parameters for data generation
noise_type = 'constant' # type of noise in labels: 'constant', 'None' (no noise), 'BTL' (Bradley-Terry-Luce model)
noise_level = 0.1 # level of noise for noise_type = 'constant'
connectivity ='connected' # graph connective enforced: 'no', 'connected', 'strongly'
bt_interval = None # scores interval for BTL model

# Parameters for PD-Rank (additional parameters can be consulted in the PDRank class. We recommend using the default values)
epsilon_out = 1e-2 # epsilon for stopping criterion of outer iterations. Recommended values of 1e-2 for m<500, 1e-1 otherwise
epsilon_in  = 1e-2 # epsilon for stopping criterion of inner iterations. Recommended values of 1e-2 for m<500, 1e-3 otherwise

##############################################################################
# RUN
##############################################################################

# Generate comparison data with the required parameters
data = Data(m,n,noise_type=noise_type,noise_level=noise_level, connectivity=connectivity,
                        bt_interval = bt_interval)
            
# If the connectivity is not None, the generated data may have a different number of items and comparisons
real_m, real_n = data.m, data.n 
x_gt = data.x # ground truth item scores
x_gt_ranking = np.argsort(x_gt) # ground truth ranking

# Data may be obtained in different formats (see methods in data_handle.py)
# PD-Rank requires a matrix A with n rows and m columns, where n is the number of comparisons and m is the number of items.
# Each row indicates a comparison between two items: with ones in the columns of the items being compared and zeros elsewhere.
# The weights w_obs are the observed weights of the comparisons. I.e., the number of times each comparison was observed.
# This only works if annotated behaviour is not included, as all observations of the same pair are aggregated for speed purposes.
# If the annotated behaviour is desired, a different method must be implemented.
A_unique, w_obs = data.get_Areduced_w()

pdrank = PDRANK(epsilon_in=epsilon_in, epsilon_out=epsilon_out)
x = pdrank.solve(A_unique, w_obs = w_obs) # solve the problem
x_pdrank_ranking = np.argsort(x) # ranking obtained by pdrank

print(f"Ground truth ranking: {x_gt_ranking}")
print(f"Retrieved ranking: {x_pdrank_ranking}")
